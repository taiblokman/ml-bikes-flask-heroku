import glob
import os
import time
import flask
import pickle
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')
# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# use pickle to load in the pre-trained model
with open(f'model/bike_model_linear.pkl', 'rb') as g:
    model_linear = pickle.load(g)
    model_name_linear = 'Linear Regression'
with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
    model_xgb = pickle.load(f)
    model_name_xgb = 'XGBoost'

### Another method of loading using pickle
# model_xgb = pickle.load(open('model/bike_model_xgboost.pkl', 'rb'))
# model_name_xgb = 'XGBoost'
# model_linear = pickle.load(open('model/bike_model_linear.pkl', 'rb'))
# model_name_linear = 'Linear'

model_name = model_name_linear

@app.route('/', methods=['GET', 'POST'])
def main():
    
    if flask.request.method == 'GET':
        # model_name = 'XGBoost'
        return(flask.render_template('main.html', model_name=model_name))

    if flask.request.method == 'POST': 
        form_name = flask.request.form['form-name']      
        form_model = flask.request.form['model-name']        

        # change classifiers
        if form_name == 'form-model':
            if form_model == 'XGBoost':   
                model_name_new = model_name_linear             
                return(flask.render_template('main.html', model_name=model_name_new))
            elif form_model == 'Linear Regression':          
                model_name_new = model_name_xgb      
                return(flask.render_template('main.html', model_name=model_name_new))
        # Predict
        if form_name == 'form-data':
            form_model = flask.request.form['model-name']
            temperature = flask.request.form['temperature']
            humidity = flask.request.form['humidity']
            windspeed = flask.request.form['windspeed']
            input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                            columns=['temperature', 'humidity', 'windspeed'],
                                            dtype=float)  
            if form_model == 'Linear Regression':
                prediction = int(model_linear.predict(input_variables)[0])
                model_name_new = form_model
            if form_model == 'XGBoost':
                prediction = int(model_xgb.predict(input_variables)[0])
                model_name_new = form_model             
            return flask.render_template('main.html',
                                        original_input={'Temperature':temperature,
                                                        'Humidity':humidity,
                                                        'Windspeed':windspeed},
                                        result=prediction,
                                        model_name=model_name_new,
                                    )
# if __name__== '__main__':
#     app.run(host='0.0.0.0',port=5000)

# Get the uploaded files
@app.route("/upload", methods=['GET','POST'])
def upload():
    # filenames = os.listdir('static/files')
    # list files in directory but remove hidden files
    # filenames = glob.glob(app.config['UPLOAD_FOLDER']+'/*')
    filenames = get_files_in_dir(app.config['UPLOAD_FOLDER'])    
    # get file time since epoch
    print(filenames)
    files_w_dates = populate_files_w_dates(filenames)

    if flask.request.method == 'GET':
        return(flask.render_template('upload.html', files=files_w_dates))
      # get the uploaded file

    if flask.request.method == 'POST':  
      uploaded_file = flask.request.files['file']
      if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # set the file path and save it
        uploaded_file.save(file_path)
        # parse the CSV
        # parseCSV(file_path)
        filename_no_ext = uploaded_file.filename.rsplit('.', 1)[0]
        out_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_no_ext+'_prediction.csv')
        process(file_path,out_file_path)
      return flask.redirect(flask.url_for('upload'))

# parse the CSV file
# note: for now we're expecting a correctly form file
def parseCSV(filePath):
      # CVS Column Names
    #   col_names = ['first_name','last_name','address', 'street', 'state' , 'zip']
      col_names = ['temperature', 'humidity', 'windspeed', 'count']
      # Use Pandas to parse the CSV file
      df = pd.read_csv(filePath,names=col_names, header=None)
      print(df.head())
      #Saving to a database, Loop through the Rows
    #   for i,row in df.iterrows():
    #          sql = "INSERT INTO addresses (first_name, last_name, address, street, state, zip) VALUES (%s, %s, %s, %s, %s, %s)"
    #          value = (row['first_name'],row['last_name'],row['address'],row['street'],row['state'],str(row['zip']))
    #          mycursor.execute(sql, value, if_exists='append')
    #          mydb.commit()
    #          print(i,row['first_name'],row['last_name'],row['address'],row['street'],row['state'],row['zip'])

def get_files_in_dir(dir):
    return [file for file in os.listdir(dir) if not file.startswith('.')]

def populate_files_w_dates(filenames):
    all_files = []
    for fls in filenames:
        # print(fls)
        modTimesinceEpoc = os.path.getmtime(app.config['UPLOAD_FOLDER']+'/'+fls)
        # Convert seconds since epoch to readable timestamp
        ep_time = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc)))
        # all dates to list of files
        all_files.append([fls,ep_time])        
        # sort by dates using lambda/tuple
        # https://stackoverflow.com/questions/65790036/python-sort-list-of-lists-by-not-one-but-two-criteria
        sorted_all_files = sorted(all_files, key=lambda x:(str(x[1]),x[0]))
    return sorted_all_files  

def process(inPath, outPath):
  # read input file
    print(inPath)
    col_names = ['temperature', 'humidity', 'windspeed']
    # Use Pandas to parse the CSV file
    # input_df = pd.read_csv(inPath,names=col_names, header=None)
    input_df = pd.read_csv(inPath)
    
    # predict the classes
    predictions = model_xgb.predict(input_df[col_names])
    print(predictions)
    # convert output labels to categories
    input_df['count'] = predictions
    # save results to csv
      # save results to csv
    # output_df = input_df[['id', 'category']]
    # output_df.to_csv(outPath, index=False)
    output_df = input_df
    output_df.to_csv(outPath, index=False)

### download files
# https://stackoverflow.com/questions/67373819/python-flask-app-which-returns-list-of-files
@app.route('/download')
def files():
    # filenames = os.listdir('static/files')
    # filenames = glob.glob(app.config['UPLOAD_FOLDER'], '*')
    filenames = get_files_in_dir(app.config['UPLOAD_FOLDER'])
    files_w_dates = populate_files_w_dates(filenames)
    return flask.render_template('files.html', files=files_w_dates)

@app.route('/files/<path:filename>')
def file(filename):
    return flask.send_from_directory(
        os.path.abspath(app.config['UPLOAD_FOLDER']),
        filename,
        as_attachment=True
    )    

if (__name__ == "__main__"):
    app.run(port = 5000)