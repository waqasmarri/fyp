from flask import Flask
app = Flask('app')
@app.route('/', methods=['GET'])
def test():
    return 'My First Flask Application, Alhamdulillah very good Application!!'

if __name__ == '__main__':
    app.run(debug=True, host='127.1.1.1', port=9000)