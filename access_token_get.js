var request = require('request');
var client_id = "76915a0c79a34e108e08ab4ca0e2605f";
var client_secret = "f30900615bad41d0ae950d4e2ad72668";

var authOptions = {
  url: 'https://accounts.spotify.com/api/token',
  headers: {
    'Authorization': 'Basic ' + (new Buffer.from(client_id + ':' + client_secret).toString('base64'))
  },
  form: {
    grant_type: 'client_credentials'
  },
  json: true
};

request.post(authOptions, function(error, response, body) {
  if (!error && response.statusCode === 200) {
    var token = body.access_token;
    console.log("Access token:", token);
  } else {
    console.error("Error:", error || body); // Output the error message or response body
  }
});
