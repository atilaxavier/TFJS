var net;
var stream;
var constraints = {
  video: { 
      facingMode: "environment",
      width: { 
          ideal: 480 
      },
      height: { 
          ideal: 640 
      }
  }
};
//webcamElement.width = 300;
//webcamElement.height = 300;

async function app() {
  var webcamElement = document.getElementById('webcam');
  console.log('Loading mobilenet..');
  var stat = document.getElementsByClassName("statusApp");
  stat[0].innerText="Loading mobilenet.";

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');
  stat[0].innerText="mobilenet loaded. Getting Video Camera.";

  stream = await navigator.mediaDevices.getUserMedia(constraints);
  stat[0].innerText="Got Video Camera. Loading webcam to TF.";
  
  webcamElement.srcObject = stream;
  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  var webcam = await tf.data.webcam(webcamElement);
  stat[0].innerText="Webcam Loaded. Capturing images.";
    
  while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);

    const res = document.getElementsByClassName("predInfo");
    res[0].innerText=result[0].className;
    const res2 = document.getElementsByClassName("probInfo");
    res2[0].innerText=result[0].probability;

    //document.getElementById('console').innerText = `
    //  prediction: ${result[0].className}\n
    //  probability: ${result[0].probability}
    //`;
    //console.log(result[0]);
    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

app();