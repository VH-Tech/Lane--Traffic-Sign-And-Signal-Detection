from flask import Flask, render_template, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from present import run
from werkzeug.utils import secure_filename
import os

from wtforms.validators import InputRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])


def home():
    form = UploadFileForm()
    
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        file.save(file_path) # Then save the file
        run(file_path)
        return "Thank You"
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)