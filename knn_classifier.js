const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
let net;
const constraints = {
        facingMode: "environment"
    }; 

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.

  const webcam = await tf.data.webcam(webcamElement, constraints);
  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, true);

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };
  
    // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  const load = async Cfier => {
     //can be change to other source
    let dataset = localStorage.getItem("myData")
    let tensorObj = JSON.parse(dataset)
    //covert back to tensor
    Object.keys(tensorObj).forEach((key) => {
      tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1024, 1024])
    });
    classifier.setClassifierDataset(tensorObj);
    console.log('Loaded classifier with ' + classifier.getNumClasses() + ' classes and ' + classifier.getNumExamples() + ' Examples');
  };

  document.getElementById('load').addEventListener('click', () => load(classifier));


  const save = async Cfier => {
    let dataset = Cfier.getClassifierDataset()
    var datasetObj = {}
        Object.keys(dataset).forEach((key) => {
             let data = dataset[key].dataSync();
             // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...] 
             // instead of object e.g {0:"0.1", 1:"-0.2"...}
             datasetObj[key] = Array.from(data); 
    });
    let jsonStr = JSON.stringify(datasetObj)
    //can be change to other source
    localStorage.setItem("myData", jsonStr);
    console.log('Saved classifier with ' + classifier.getNumClasses() + ' classes and ' + classifier.getNumExamples() + ' Examples');
  };

  document.getElementById('save').addEventListener('click', () => save(classifier));

  const clear = async () => {
        localStorage.clear();
    console.log('Clear LocalStorage');
  };

  document.getElementById('clear').addEventListener('click', () => clear(classifier));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      // Dispose the tensor to release the memory.
      img.dispose();
    }
    await tf.nextFrame();
  }
}


app();