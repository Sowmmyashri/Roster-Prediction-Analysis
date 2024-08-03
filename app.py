import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, render_template, request, redirect, url_for, flash, make_response,session,send_file,send_from_directory
from flask_socketio import SocketIO, emit
import os

app = Flask(__name__)

app.secret_key = 'hello'
socketio = SocketIO(app)

SWAP_FILES_PATH = r'D:\Virtusa\Swap prediction'
LEAVE_FILES_PATH = r'D:\Virtusa\Leave Prediction'
UPLOAD_FOLDER = r'C:\Users\tsowm\Downloads'
LOGIN_FILE_PATH = r'D:\Virtusa\User Credentials.xlsx' 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_user(username, password):
    df = pd.read_excel(LOGIN_FILE_PATH)
    user = df[df['Username'] == username]
    if not user.empty and user['Password'].values[0] == password:
        return True
    return False

def rewrite_excel(file_path):
    df = pd.read_excel(file_path)
    
    # Fill forward missing values in the 'Month' and 'Year' columns
    df['Month'].fillna(method='ffill', inplace=True)
    df['Year'].fillna(method='ffill', inplace=True)
    
    # Convert 'Month' and 'Year' to integers
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(int)
    
    # Identify the current month and year
    current_month = df['Month'].iloc[0]
    current_year = df['Year'].iloc[0]
    
    # Calculate the next month and year
    next_month = current_month % 12 + 1
    next_year = current_year + (1 if next_month == 1 else 0)  # Increment year if next month is January
    
    # Determine the first day of the next month
    first_day_of_next_month = datetime(next_year, next_month, 1).strftime('%a').lower()
    
    # Create a list of days in the correct order starting from the first day of the next month
    days_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    start_index = days_order.index(first_day_of_next_month)
    ordered_days = days_order[start_index:] + days_order[:start_index]
    
    # Update the month and year values
    df['Month'] = next_month
    df['Year'] = next_year
    
    # Sort the DataFrame based on the new order of days
    df['Req For'] = pd.Categorical(df['Req For'], categories=ordered_days, ordered=True)
    df = df.sort_values('Req For').reset_index(drop=True)
    
    # Save the updated DataFrame back to the same file path
    df.to_excel(file_path, index=False)

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', title='Admin Portal Home')
    return redirect(url_for('login'))

@app.route('/base')
def base():
    if 'username' in session:
        return render_template('base.html', title='Admin Portal Home')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if validate_user(username, password):
            session['username'] = username
            if username == 'Admin':
                return redirect(url_for('base'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/')
def ben():
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/admin_home')
def admin_home():
    if 'username' in session:
        return render_template('admin_home.html', title='Home')

@app.route('/statistics')
def statistics():
    return render_template('base.html', page='statistics')

def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/swap_rate_prediction')
def swap_rate_prediction():
    # Read the training data
    df = pd.read_excel(os.path.join(SWAP_FILES_PATH, 'Swaps.xlsx'))
    # Read the test data
    testdf = pd.read_excel(os.path.join(SWAP_FILES_PATH, 'testswap.xlsx'))

    # Encode categorical variables
    label_encoder_shift = LabelEncoder()
    df['Shift_Of_Requester'] = label_encoder_shift.fit_transform(df['Shift_Of_Requester'])
    df['Requested_Shift'] = label_encoder_shift.fit_transform(df['Requested_Shift'])

    # Prepare training data
    X = df[['Requester_ID', 'Recipient_ID', 'Shift_Of_Requester', 'day']]
    y = df['Requested_Shift']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Train the model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=label_encoder_shift.classes_)

    # Encode the test data's 'Shift_Of_Requester' column
    testdf['Shift_Of_Requester'] = label_encoder_shift.transform(testdf['Shift_Of_Requester'])

    # Make predictions on the test data
    y_pred_test = clf.predict(testdf[['Requester_ID', 'Recipient_ID', 'Shift_Of_Requester', 'day']])

    # Convert the predicted labels back to their original form
    predicted_shifts = label_encoder_shift.inverse_transform(y_pred_test)

    # Convert 'Shift_Of_Requester' column back to original form for display
    testdf['Shift_Of_Requester'] = label_encoder_shift.inverse_transform(testdf['Shift_Of_Requester'])

    # Add the predicted shifts to the test DataFrame
    testdf['Predicted_Shift'] = predicted_shifts

    # Display the DataFrame with predicted shifts
    swap_prediction_result = testdf[['Requester_Name', 'Recipient_Name', 'Shift_Of_Requester', 'Predicted_Shift']].to_html(index=False)

    # Analysis of Most Requested and Most Swapped Shifts
    unique_employee_ids = sorted(df['Requester_ID'].unique())

    requested_shifts_results = []
    swapped_shifts_results = []

    def find_requested_shifts(employee_id, data):
        employee_data = data[data['Requester_ID'] == employee_id]
        shift_counts = employee_data['Requested_Shift'].value_counts()
        total_shifts = shift_counts.sum()
        shift_percentages = (shift_counts / total_shifts) * 100
        return shift_counts.index.tolist(), shift_percentages.tolist()

    def find_swapped_shifts(employee_id, data):
        employee_data = data[data['Requester_ID'] == employee_id]
        shift_counts = employee_data['Shift_Of_Requester'].value_counts()
        total_shifts = shift_counts.sum()
        shift_percentages = (shift_counts / total_shifts) * 100
        return shift_counts.index.tolist(), shift_percentages.tolist()

    for employee_id in unique_employee_ids:
        requested_shifts, percentages = find_requested_shifts(employee_id, df)
        requested_shifts_results.extend(zip([employee_id] * len(requested_shifts), requested_shifts, percentages))
        
        swapped_shifts, percentages = find_swapped_shifts(employee_id, df)
        swapped_shifts_results.extend(zip([employee_id] * len(swapped_shifts), swapped_shifts, percentages))

    requested_shifts_df = pd.DataFrame(requested_shifts_results, columns=['Employee Id', 'Most Requested Shift', 'Shift Request Probability']).sort_values(by='Employee Id')
    swapped_shifts_df = pd.DataFrame(swapped_shifts_results, columns=['Employee Id', 'Most Swapped Shift', 'Shift Swap Probability']).sort_values(by='Employee Id')

    # Define the mapping of shift codes to their alphabetic representations
    shift_mapping = {0: 'M', 1: 'A', 2: 'N', 3: 'O'}

    # Map the shift codes to their alphabetic representations in the dataframes
    requested_shifts_df['Most Requested Shift'] = requested_shifts_df['Most Requested Shift'].map(shift_mapping)
    swapped_shifts_df['Most Swapped Shift'] = swapped_shifts_df['Most Swapped Shift'].map(shift_mapping)

    # Plotting Most Requested and Most Swapped Shifts
    for employee_id in unique_employee_ids:
        employee_data = df[df['Requester_ID'] == employee_id]

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot most requested shifts
        requested_shift_counts = employee_data['Requested_Shift'].value_counts()
        total_requested_shifts = requested_shift_counts.sum()
        requested_shift_percentages = (requested_shift_counts / total_requested_shifts) * 100
        requested_shift_percentages.index = requested_shift_percentages.index.map(shift_mapping)
        requested_shift_percentages.plot(kind='bar', color='skyblue', ax=ax1)
        ax1.set_title(f"Most Requested Shifts for Employee {employee_id}")
        ax1.set_xlabel("Shift")
        ax1.set_ylabel("Percentage")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        # Plot most swapped shifts
        swapped_shift_counts = employee_data['Shift_Of_Requester'].value_counts()
        total_swapped_shifts = swapped_shift_counts.sum()
        swapped_shift_percentages = (swapped_shift_counts / total_swapped_shifts) * 100
        swapped_shift_percentages.index = swapped_shift_percentages.index.map(shift_mapping)
        swapped_shift_percentages.plot(kind='bar', color='salmon', ax=ax2)
        ax2.set_title(f"Most Swapped Shifts for Employee {employee_id}")
        ax2.set_xlabel("Shift")
        ax2.set_ylabel("Percentage")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the combined figure
        plt.savefig(f'static/shift_stats_{employee_id}.png')
        plt.close()  # Close the plot after saving

    response = make_response(render_template('result.html', 
                           title='Swap Rate Prediction Result', 
                           accuracy=accuracy, 
                           conf_matrix=conf_matrix.tolist(), 
                           class_report=class_report,
                           requested_shifts_table=requested_shifts_df.to_html(index=False),
                           swapped_shifts_table=swapped_shifts_df.to_html(index=False),
                           unique_employee_ids=unique_employee_ids,
                           page='swap_rate_prediction'))
    return no_cache(response)

@app.route('/effort_rate_prediction')
def effort_rate_prediction():
    # Read Excel files for each month
    df_jan = pd.read_excel(os.path.join(LEAVE_FILES_PATH, 'Jan_Leave.xlsx'))
    df_feb = pd.read_excel(os.path.join(LEAVE_FILES_PATH, 'Feb_Leave.xlsx'))
    df_mar = pd.read_excel(os.path.join(LEAVE_FILES_PATH, 'March_Leave.xlsx'))
    df_apr = pd.read_excel(os.path.join(LEAVE_FILES_PATH, 'April_Leave.xlsx'))
    df_may = pd.read_excel(os.path.join(LEAVE_FILES_PATH, 'May_Leave.xlsx'))

    # Number of days in each month
    days_in_month = {'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30, 'May': 31}

    # Combine all dataframes into a single dataframe
    df_combined = pd.concat([df_jan, df_feb, df_mar, df_apr, df_may], ignore_index=True)

    # Function to calculate the attendance percentage for each month
    def calculate_attendance_percentage(df):
        df['Total_Leaves'] = df.iloc[:, 1:].replace(-1, pd.NA).count(axis=1)
        df['Attendance_Percentage'] = ((df['Total_Leaves'].rsub(days_in_month['May'])) / days_in_month['May']) * 100
        return df[['Emp_Id', 'Attendance_Percentage']]

    # Calculate attendance percentages for all months
    df_combined = calculate_attendance_percentage(df_combined)

    # Prepare the feature matrix X and target vector y
    X = df_combined[['Emp_Id']]  
    y = df_combined['Attendance_Percentage']

    # Initialize the Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor(random_state=42)

    # Train the model
    gb_regressor.fit(X, y)

    # Prepare data for predicting the attendance percentages for May
    X_may = df_may[['Emp_Id']]  

    # Predict attendance percentages for May
    y_pred_may = gb_regressor.predict(X_may)

    # Create a new dataframe with predictions for May
    df_pred_may = pd.DataFrame({'Emp_Id': X_may['Emp_Id'], 'May_Attendance_Percentage': y_pred_may})

    # Find the employee with the highest predicted attendance percentage for May
    most_attentive_employee = df_pred_may.loc[df_pred_may['May_Attendance_Percentage'].idxmax()]

    # Plot the predicted attendance percentages for May
    plt.figure(figsize=(10, 5))
    plt.bar(df_pred_may['Emp_Id'], df_pred_may['May_Attendance_Percentage'], color='purple')
    plt.xlabel('Employee ID')
    plt.ylabel('Attendance Percentage (%)')
    plt.title('Predicted Attendance Percentages for May')
    plt.xticks(df_pred_may['Emp_Id'])
    plt.grid(True)

    # Dynamically adjust y-axis limits
    plt.ylim(min(df_pred_may['May_Attendance_Percentage']) - 5, 100)

    # Ensure 'static' directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Save the plot
    plt.savefig('static/predicted_attendance.png')

    effort_prediction_result = df_pred_may.to_html(index=False)

    # Prepare the response
    response = make_response(render_template('result.html', 
                           title='Effort Rate Prediction Result', 
                           table=effort_prediction_result, 
                           most_attentive_employee=most_attentive_employee.to_dict(),
                           plot_url='/static/predicted_attendance.png',
                           page='effort_rate_prediction'))
    return no_cache(response)

@app.route('/demand', methods=['GET', 'POST'])
def demand():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            session['file_uploaded'] = True
            session['uploaded_filename'] = file.filename
            rewrite_excel(file_path)
            
            df = pd.read_excel(file_path)
            file_content = df.to_html(classes='table table-striped', index=False)
            
            flash('File uploaded successfully.')
            session['file_content'] = file_content  # Save the content to session
            
            return render_template('upload.html', page='demand', file_uploaded=True, file_content=file_content)
        else:
            flash('Invalid file type! Only .xlsx files are allowed.')
            return redirect(request.url)
    
    if session.get('file_uploaded'):
        file_content = session.get('file_content', '')
        return render_template('upload.html', page='demand', file_uploaded=True, file_content=file_content)
    
    return render_template('upload.html', page='demand', file_uploaded=False, file_content='')

@app.route('/save_changes', methods=['POST'])
def save_changes():
    edited_table_html = request.form['file_content']
    df = pd.read_html(edited_table_html)[0]  # Read HTML table back to DataFrame
    
    # Get the original file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], session.get('uploaded_filename'))
    
    # Save the updated DataFrame back to Excel file
    df.to_excel(file_path, index=False)
    
    # Update session with new file content
    session['file_content'] = edited_table_html
    
    # Emit SocketIO event to notify clients of the update
    socketio.emit('update_table', {'file_content': edited_table_html}, namespace='/file_updates')
    
    flash('Changes saved successfully.')
    return redirect(url_for('demand'))

@app.route('/get_latest_file')
def get_latest_file():
    # Replace with your actual file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], session.get('uploaded_filename', ''))
    
    # Get the last modified time of the file
    last_modified = os.path.getmtime(file_path)
    
    if 'last_modified' not in session or session['last_modified'] < last_modified:
        session['last_modified'] = last_modified
        
        # Read the updated content from the file
        df = pd.read_excel(file_path)
        file_content = df.to_html(classes='table table-striped', index=False)
        
        # Emit an event to update the table content
        socketio.emit('update_table', {'file_content': file_content}, namespace='/file_updates')
    
    return send_file(file_path)

file_content = ""

@socketio.on('update_file_content', namespace='/file_updates')
def update_file_content(data):
    global file_content
    file_content = data['file_content']
    emit('update_table', {'file_content': file_content}, broadcast=True)


@socketio.on('connect', namespace='/file_updates')
def handle_connect():
    emit('update_table', {'file_content': file_content})


if __name__ == '__main__':
    socketio.run(app, debug=True)
    