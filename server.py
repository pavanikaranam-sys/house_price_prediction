from flask import Flask, jsonify
import util

app=Flask(__name__)

@app.route('/get_location_names')

def get_location_names():
    response=jsonify({
        'locations':util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response
    
    return "this is new flask app"

if __name__=='__main__':
    app.run(debug=True)
