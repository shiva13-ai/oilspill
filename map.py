from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
from anamoly import process_ais_data

UPLOAD_FOLDER = os.path.join('static', 'uploads')

# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/working')
def working():
      return render_template('working.html')

# Route for About Page
@app.route('/about')
def about():
    return render_template('about.html')


# Route for Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')


# Route for Feature Page
@app.route('/feature')
def feature():
    return render_template('feature.html')


# Route for Pricing Page
@app.route('/price')
def price():
    return render_template('price.html')
@app.route('/team')
def team():
      return render_template('team.html')

# Route for Quote Page
@app.route('/quote')
def quote():
    return render_template('quote.html')

@app.route('/uploadFile', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'number' in request.form:
            try:
                # Convert the form input to an integer
                number = int(request.form['number'])
            except ValueError:
                # Handle cases where the input cannot be converted to an integer
                return "Invalid input! Please enter a valid integer.", 400
            file_path=session.get('file_path')
            result=process_ais_data(file_path,number)
            return render_template('working.html',result=result)
        # Get the file from the request
        file = request.files['file']

        # Save the file without any validation
        data_filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        session['file_path']=file_path
        # Ensure the directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the file
        file.save(file_path)

        # Store the uploaded file path in the session
        session['uploaded_data_file_path'] = file_path

        return render_template('working.html', file_name=data_filename)

    return render_template('index.html')


@app.route('/show_data')
def showData():
    # Get the uploaded file path from the session
    data_file_path = session.get('uploaded_data_file_path', None)

    if not data_file_path or not os.path.exists(data_file_path):
        return render_template('error.html', message="No file uploaded or file not found.")

    try:
        # Read the CSV file
        uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')

        # Convert the dataframe to an HTML table
        uploaded_df_html = uploaded_df.to_html()

        return render_template('show_csv_data.html', data_var=uploaded_df_html)
    except Exception as e:
        # Handle errors in reading the CSV file
        return render_template('error.html', message=f"Error reading the file: {str(e)}")


if __name__ == '__main__':
	app.run(debug=True)
