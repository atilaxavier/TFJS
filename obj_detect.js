let net;

const img = document.getElementById('img');

async function app() {
    console.log('Loading cocoSsd..');
    // Load the model.
    net = await cocoSsd.load();
    console.log('Successfully loaded model');

    const predictions = await net.detect(img);
    console.log('Predictions: ', predictions);
    
//    cocoSsd.load().then(model => {
        // detect objects in the image.
//        model.detect(img).then(predictions => {
//          console.log('Predictions: ', predictions);
//        });
//    });
}

app();