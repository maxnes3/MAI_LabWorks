from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
from num2words import num2words

app = Flask(__name__)


def load_data(filename):
    return pd.read_csv(filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/display_data', methods=['POST'])
def display_data():
    csv_data = load_data('data.csv')

    start_row = int(request.form['start_row'])
    end_row = int(request.form['end_row'])
    start_col = int(request.form['start_col'])
    end_col = int(request.form['end_col'])

    if(start_col < 1 or start_col > len(csv_data.columns) or
    end_col < 1 or end_col > len(csv_data.columns) or
    start_row < 1 or start_row > len(csv_data) or
    end_row < 1 or end_row > len(csv_data)):
        return render_template('error_data.html',
                               error_info='Невернные данные')

    # Выбор диапазона данных
    selected_data = csv_data.iloc[start_row-1:end_row, start_col-1:end_col]

    # Описание столбцов
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

    # Описание набора данных
    dataset_description = {
        'total_rows': len(selected_data),
        'total_cols': len(selected_data.columns),
        'num_empty_cells': selected_data.isna().sum().sum(),
        'num_filled_cells': selected_data.count().sum()
    }

    # Генерация описания столбцов и набора данных на основе num2words
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
        dataset_description=dataset_description
    )


@app.route('/download', methods=['GET'])
def download_file():

    # Отправьте файл на клиентскую сторону
    return send_file('data.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)