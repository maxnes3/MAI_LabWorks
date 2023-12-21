from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
from num2words import num2words
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from BloomFilter import BloomFilter
from PairedRegression import PairedRegression
from DecisionTree import DecisionTree

app = Flask(__name__)


def load_data(filename):
    return pd.read_csv(filename)


def throw_error(message):
    return render_template('error_data.html', error_info=message)


@app.route('/')
def index():
    csv_data = load_data('stroke_data.csv')
    numeric_columns = csv_data.select_dtypes(include='number').columns.tolist()
    return render_template('index.html',
                           max_row=len(csv_data),
                           max_col=len(csv_data.columns),
                           columns=csv_data.columns,
                           numeric_columns=numeric_columns)


@app.route('/display_data', methods=['POST'])
def display_data():
    csv_data = load_data('stroke_data.csv')

    start_row = int(request.form['start_row'])
    end_row = int(request.form['end_row'])
    start_col = int(request.form['start_col'])
    end_col = int(request.form['end_col'])

    if (start_col < 1 or start_col > len(csv_data.columns) or
            end_col < 1 or end_col > len(csv_data.columns) or
            start_row < 1 or start_row > len(csv_data) or
            end_row < 1 or end_row > len(csv_data)):
        return throw_error('Невернные данные')

    selected_data = csv_data.iloc[start_row - 1:end_row, start_col - 1:end_col]

    column_descriptions = []
    for col in selected_data.columns:
        col_data = selected_data[col]
        col_type = col_data.dtype
        num_empty_cells = col_data.isna().sum()
        num_filled_cells = col_data.count()
        col_description = {
            'name': col,
            'type': col_type,
            'empty_cells': num_empty_cells,
            'filled_cells': num_filled_cells
        }
        column_descriptions.append(col_description)

    dataset_description = {
        'total_rows': len(selected_data),
        'total_cols': len(selected_data.columns),
        'num_empty_cells': selected_data.isna().sum().sum(),
        'num_filled_cells': selected_data.count().sum()
    }

    for col_desc in column_descriptions:
        col_desc['empty_cells_words'] = num2words(col_desc['empty_cells'], lang='ru')
        col_desc['filled_cells_words'] = num2words(col_desc['filled_cells'], lang='ru')

    dataset_description['total_rows_words'] = num2words(dataset_description['total_rows'], lang='ru')
    dataset_description['total_cols_words'] = num2words(dataset_description['total_cols'], lang='ru')
    dataset_description['num_empty_cells_words'] = num2words(dataset_description['num_empty_cells'], lang='ru')
    dataset_description['num_filled_cells_words'] = num2words(dataset_description['num_filled_cells'], lang='ru')

    return render_template(
        'display_data.html',
        selected_data=selected_data.to_html(classes='table table-dark table-bordered table-hover'),
        column_descriptions=column_descriptions,
        dataset_description=dataset_description,
        columns=selected_data.columns
    )


@app.route('/analysis_data', methods=['POST'])
def analysis_data():
    csv_data = load_data('stroke_data.csv')

    selected_condition = str(request.form['selected_condition'])
    condition_value = str(request.form['condition_value'])
    selected_column = str(request.form['selected_column'])

    filtered_df = csv_data.groupby(selected_condition)

    if not (condition_value in filtered_df.groups):
        return throw_error('Значения нету в колонке')

    filtered_df = filtered_df.get_group(condition_value)

    min_value = filtered_df[selected_column].min()
    mean_value = filtered_df[selected_column].mean()
    max_value = filtered_df[selected_column].max()

    return render_template('analysis_data.html',
                           min_value=round(min_value, 2),
                           mean_value=round(mean_value, 2),
                           max_value=round(max_value, 2),
                           column_name=selected_column,
                           filtered_df=filtered_df.to_html(classes='table table-dark table-bordered table-hover'))


@app.route('/graphics_data', methods=['POST'])
def graphics_data():
    csv_data = load_data('stroke_data.csv')

    selected_condition = str(request.form['selected_condition'])
    condition_value = str(request.form['condition_value'])
    selected_column = str(request.form['selected_column'])

    extended_data = csv_data

    filtered_df = csv_data.groupby(selected_condition)

    if not (condition_value in filtered_df.groups):
        return throw_error('Значения нету в колонке')

    filtered_df = filtered_df.get_group(condition_value)

    newrow_count = int(len(csv_data) * 0.1)
    for i in range(newrow_count):
        new_row = {}
        for col in csv_data.columns:
            if csv_data[col].dtype in [int, float]:
                new_row[col] = csv_data[col].mean()
            else:
                new_row[col] = csv_data[col].mode().iloc[0]
        new_data = pd.DataFrame(new_row, index=[len(extended_data)])
        extended_data = pd.concat([extended_data, new_data], ignore_index=True)

    filtered_ef = extended_data.groupby(selected_condition)
    filtered_ef = filtered_ef.get_group(condition_value)

    diagram_data = {'Data1': [filtered_df[selected_column].min(), filtered_df[selected_column].mean(),
                              filtered_df[selected_column].max()],
                    'Data2': [filtered_ef[selected_column].min(), filtered_ef[selected_column].mean(),
                              filtered_ef[selected_column].max()],
                    'Labels': ['Min', 'Average', 'Max']}

    df = pd.DataFrame(diagram_data)

    fig, ax = plt.subplots()
    width = 0.3
    x = range(len(df))
    ax.bar(x, df['Data1'], width, label='CSV-файл')
    ax.bar([i + width for i in x], df['Data2'], width, label='CSV-файл + 10%')
    ax.set_xlabel('Показатели')
    ax.set_ylabel('Значения')
    ax.set_title('Сравнение данных')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(df['Labels'])
    ax.legend()

    for i, v1, v2 in zip(x, df['Data1'], df['Data2']):
        ax.text(i - 0.1, v1 + 1, str(round(v1, 2)), color='black', fontweight='bold')
        ax.text(i + width - 0.1, v2 + 1, str(round(v2, 2)), color='black', fontweight='bold')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.read()).decode()

    return render_template('graphics_data.html',
                           plot_url=plot_url,
                           column_name=selected_column)


@app.route('/bloomfilter_data', methods=['POST'])
def bloomfilter_data():
    search_words = str(request.form['search_words']).split()

    name_href_1 = ['Stroke Prediction Dataset',
                   'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset',
                   False]
    name_href_2 = ['Supermarket store branches sales analysis',
                   'https://www.kaggle.com/datasets/surajjha101/stores-area-and-sales-data',
                   False]
    name_href_3 = ['NASA - Nearest Earth Objects',
                   'https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects',
                   False]

    colums_1 = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'Residence_type', 'avg_glucose_level', 'bmi']
    colums_2 = ['Store ID', 'Store_Area', 'Items_Available', 'Store_Sales']
    colums_3 = ['id', 'name', 'est_diameter_min', 'relative_velocity', 'miss_distance',
                'orbiting_body', 'sentry_object', 'hazardous']

    bloomFilter_1 = BloomFilter(colums_1, 400, 4)
    bloomFilter_2 = BloomFilter(colums_2, 400, 4)
    bloomFilter_3 = BloomFilter(colums_3, 400, 4)

    output_matrix = []

    for item in search_words:
        output_matrix.append({
            'name': item,
            'check_1': bloomFilter_1.check(item),
            'check_2': bloomFilter_2.check(item),
            'check_3': bloomFilter_3.check(item)})

        if bloomFilter_1.check(item):
            name_href_1[2] = bloomFilter_1.check(item)

        if bloomFilter_2.check(item):
            name_href_2[2] = bloomFilter_2.check(item)

        if bloomFilter_3.check(item):
            name_href_3[2] = bloomFilter_3.check(item)

    return render_template('bloomfilter_data.html',
                           name_1=name_href_1[0], href_1=name_href_1[1], check_1=name_href_1[2],
                           name_2=name_href_2[0], href_2=name_href_2[1], check_2=name_href_2[2],
                           name_3=name_href_3[0], href_3=name_href_3[1], check_3=name_href_3[2],
                           output_matrix=output_matrix)


@app.route('/regression_data', methods=['POST'])
def regression_data():
    csv_data = load_data('stroke_data.csv')

    selected_column1 = str(request.form['selected_column1'])
    selected_column2 = str(request.form['selected_column2'])

    if selected_column1 == selected_column2:
        return throw_error("Нужно выбрать разные колонки")

    big_data = csv_data.sample(frac=0.99)
    small_data = csv_data.drop(big_data.index)

    model = PairedRegression(big_data, selected_column1, selected_column2)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.scatter(big_data[selected_column1].values, big_data[selected_column2].values, alpha=0.4)
    plt.plot(big_data[selected_column1].values, model.predict(big_data[selected_column1].values),
             color='red', linewidth=3)
    plt.xlabel(selected_column1)
    plt.ylabel(selected_column2)

    plt.subplot(2, 1, 2)
    plt.scatter(small_data[selected_column1].values, small_data[selected_column2].values, alpha=0.4)
    plt.plot(small_data[selected_column1].values, model.predict(small_data[selected_column1].values),
             color='red', linewidth=3)
    plt.xlabel(selected_column1)
    plt.ylabel(selected_column2)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    graph_url = base64.b64encode(img.read()).decode()

    return render_template("regression_data.html",
                           graph_url=graph_url,
                           selected_column1=selected_column1,
                           selected_column2=selected_column2)


@app.route("/decisiontree_data", methods=['GET'])
def decisiontree_data():
    csv_data = load_data('stroke_data.csv')
    csv_data = csv_data.dropna()

    csv_data['gender'] = csv_data['gender'].map({'Male': 0, 'Female': 1})

    learn_rows = csv_data.sample(n=25, random_state=42)
    test_rows = csv_data.sample(n=5, random_state=42)

    tree = DecisionTree(min_samples=2, max_depth=3, data=learn_rows.to_dict('records'))
    output_test = []
    for index, row in test_rows.iterrows():
        output_test.append({
            "column1": "age",
            "value1": row["age"],
            "column2": "gender",
            "value2": row["gender"],
            "target": row["bmi"],
            "predict": tree.getPrediction(age=row["age"], gender=row["gender"])
        })

    return render_template("decisiontree_data.html",
                           learn_rows=learn_rows.to_html(classes='table table-dark table-bordered table-hover'),
                           test_rows=test_rows.to_html(classes='table table-dark table-bordered table-hover'),
                           output_test=output_test)


@app.route('/download', methods=['GET'])
def download_file():
    return send_file('stroke_data.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
