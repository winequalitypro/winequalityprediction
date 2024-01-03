from flask import Flask, render_template, redirect, url_for,flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from database import db, User
from flask_mail import Mail, Message
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv(dotenv_path="wine.env")
secret_key = os.environ.get("SECRET_KEY")
database_uri = os.environ.get("DATABASE_URI")
mail_username=os.environ.get("MAIL_USERNAME")
mail_password=os.environ.get("MAIL_PASSWORD")
mail_default_sender=os.environ.get("MAIL_DEFAULT_SENDER")



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = secret_key  # Set a secret key for session security
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465 # Use the appropriate port for your mail server
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = mail_username
app.config['MAIL_PASSWORD'] = mail_password
app.config['MAIL_DEFAULT_SENDER'] = mail_default_sender

mail = Mail(app)
# Load the wine dataset
wine_df = pd.read_csv('winequality-red.csv')

# Create the predictor (X) and target (y) variables
X = wine_df.drop('quality', axis=1).values
y = wine_df['quality'].apply(lambda yval: 1 if yval >= 7 else 0).values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# Train the Random Forest Classifier model
model = RandomForest()
model.fit(X_train, Y_train)
# accuracy on test data
X_test_prediction = model.predict(X_test)
print(accuracy_score(X_test_prediction, Y_test))
@app.route('/predict',methods=['GET'])
@login_required
def predict_form():
    return render_template('predict.html')

@app.route('/aboutus',methods=['GET'])
@login_required
def aboutus():
    return render_template('aboutus.html')



@app.route('/contactadmin', methods=['POST','GET'])
@login_required

def send_message():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Send email
        subject = f"New Message from {name}"
        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"

        msg = Message(subject, recipients=[mail_username])
        msg.body = body

        mail.send(msg)

        return render_template('contactadmin.html')
    else:
        return render_template('contactadmin.html')




@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout successful.')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Extract features from the form fields
    features = np.array([float(request.form['fixed_acidity']),
                float(request.form['volatile_acidity']),
                float(request.form['citric_acid']),
                float(request.form['residual_sugar']),
                float(request.form['chlorides']),
                float(request.form['free_sulfur_dioxide']),
                float(request.form['total_sulfur_dioxide']),
                float(request.form['density']),
                float(request.form['pH']),
                float(request.form['sulphates']),
                float(request.form['alcohol'])])
  
    # Convert features to float and make prediction
    prediction = model.predict(features.reshape(1,-1))
    print("helllo",prediction)

   
    # Determine the result based on the prediction
    result = "Good Quality Wine" if prediction == 1 else "Bad Quality Wine"

    # Render the result in the template
    return render_template('predict.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/privacypolicy',methods=['GET'])
@login_required
def policy():
    return render_template('policy.html')

@app.route('/terms',methods=['GET'])
@login_required
def terms():
    return render_template('terms.html')







if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)
uvicorn.run(app, host="0.0.0.0", port=5000)