from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
from num2words import num2words
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)


def load_data(filename):
    return pd.read_csv(filename)


def throw_error(message):
    return render_template('error_data.html', error_info=message)


@app.route('/')
def index():
    csv_data = load_data('stroke_data.csv')
    return render_template('index.html',
                           max_row=len(csv_data),
                           max_col=len(csv_data.columns),
                           columns=csv_data.columns)


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


@app.route('/download', methods=['GET'])
def download_file():
    return send_file('stroke_data.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
