import sys
sys.path.append('/home/deepak/Desktop/Files/SEM-6/EE390/neural-machine-translation/scripts')

import marian_translate
import mytranslate

from flask import Flask, request, redirect, render_template, jsonify

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

@app.route('/')
def init():
    return render_template('index.html')

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    inp = str(request.form['text'])
    print('\n\n',inp,'\n\n')

    # outData = {'trn': marian_translate.gen_translation(inp['inp'])}

    # return json.dumps(outData, ensure_ascii=False)
    # print(jsonify({'trn': marian_translate.gen_translation(inp['inp'])}))
    
    return render_template('index.html', translation = str(marian_translate.gen_translation(inp)))

if __name__ == '__main__':
    app.run()