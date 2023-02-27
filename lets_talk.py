# Lembra do arquivo que criamos? Vamos importá-lo aqui
import pickle

# pandas serve para lidar com dados tabulares
import pandas as pd

# Flask é um microframework para criar aplicações web
from flask import Flask, render_template, request

# Instanciando a aplicação com a pasta de templates
app = Flask(__name__, template_folder='template', static_folder='template/assets')

# Treina lá, usa cá
modelo_pipeline = pickle.load(open('./models/modelo_churn.pkl', 'rb'))


# Endpoint principal (home)
@app.route('/')
def home():
    return render_template("homepage.html")


# Endpoint para o formulário
@app.route('/dados_cliente')
def dados_cliente():
    return render_template("form.html")


# Função auxiliar para lidar com o formulário
def get_data():
    # Pega os dados do formulário
    tenure = request.form.get('tenure')
    MonthlyCharges = request.form.get('MonthlyCharges')
    TotalCharges = request.form.get('TotalCharges')
    gender = request.form.get('gender')
    SeniorCitizen = request.form.get('SeniorCitizen')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    PhoneService = request.form.get('PhoneService')
    MultipleLines = request.form.get('MultipleLines')
    InternetService = request.form.get('InternetService')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = request.form.get('DeviceProtection')
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')
    Contract = request.form.get('Contract')
    PaperlessBilling = request.form.get('PaperlessBilling')
    PaymentMethod = request.form.get('PaymentMethod')

    # Cria um dicionário com os dados do formulário
    d_dict = {'tenure': [tenure], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges],
              'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner],
              'Dependents': [Dependents], 'PhoneService': [PhoneService],
              'MultipleLines': [MultipleLines], 'InternetService': [InternetService],
              'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup],
              'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport],
              'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies],
              'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
              'PaymentMethod': [PaymentMethod]}

    # Cria um dataframe com os dados do formulário
    return pd.DataFrame.from_dict(d_dict, orient='columns')


# Endpoint que executa a classificação do cliente e retorna o resultado
@app.route('/send', methods=['POST'])
def show_data():
    # Pega os dados do formulário
    df = get_data()

    # reordena as colunas (elas tem que estar na mesma ordem do modelo)
    df = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
             'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
             'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
             'MonthlyCharges', 'TotalCharges']]

    # Executa a classificação
    prediction = modelo_pipeline.predict(df)

    # Se a classificação for 0, o cliente não vai cancelar
    if prediction == 0:
        mensagem = 'Ufa..i. esse cliente vai ficar!! Aproveita pra entubar uns serviços novos!'
        imagem = 'chefe_felz.jpg'
    else:
        # Se a classificação for 1, o cliente vai sair
        mensagem = 'DANGER!!! VAI VAZAR! TELEMARKETING NELE!!'
        imagem = 'chefe_brabo.jpg'

    return render_template('result.html', tables=[df.to_html(classes='data', header=True, col_space=10)],
                           result=mensagem, imagem=imagem)


# Roda a aplicação
if __name__ == "__main__":
    app.run(debug=True)
