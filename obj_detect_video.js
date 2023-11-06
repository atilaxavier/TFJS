var net;
var webcam;
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

async function app() {
  var webcamElement = document.getElementById('webcam');
  //const webcamElement = document.querySelector('video');

  console.log('Loading cocoSsd..');
  var stat = document.getElementsByClassName("statusApp");
  stat[0].innerText="Loading cocoSSD.";

  // Load the model.
  net = await cocoSsd.load();
  //net = await cocoSsd.load('lite_mobilenet_v2');
  //console.log('Successfully loaded model');
  //const stream = await navigator.mediaDevices.getUserMedia({
  //  audio: false,
  //  video: {
        //facingMode: { exact: "environment" } 
   //     facingMode: "environment"
        //facingMode: "user"
  //      }
  //})
  stat[0].innerText="cocoSSD loaded. Getting Video Camera.";

 
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    alert('Erro no navigator.mediaDevices.getUserMedia - exception :' + e);
  };
  //const stream = await navigator.mediaDevices.getUserMedia(constraints);
  stat[0].innerText="Got Video Camera. Loading webcam to TF.";

  webcamElement.srcObject = stream;

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  try {
    webcam = await tf.data.webcam(webcamElement);
  } catch (e) {
    alert('Erro no tf.data.webcam - exception : ' + e);
  };
  
  console.log('webcamElement vw x vh: ' + webcamElement.videoWidth + " x " + webcamElement.videoHeight );
  stat[0].innerText="Webcam Loaded. Capturing images.";
  const res = document.getElementsByClassName("resolutionInfo");
  res[0].innerText=webcamElement.videoWidth + "x" + webcamElement.videoHeight;
  
  while (true) {
    const img = await webcam.capture();
    const predictions = await net.detect(img);
    //console.log('Predictions: ', predictions);
    // ##
    //const canvas = <HTMLCanvasElement> document.getElementById("canvas");
    const canvas = document.getElementById("canvas");
    
    const ctx = canvas.getContext("2d");
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight; 
    //console.log('Canvas : '+ webcamElement.videoWidth + ' x ' + webcamElement.videoHeight);    
    //canvas.width  = 300;
    //canvas.height = 300;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    //ctx.drawImage(webcamElement,0, 0,300,300);
    ctx.drawImage(webcamElement,0, 0,canvas.width,canvas.height);

    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      // prediction.score - 0.909...
      // prediction.class - "person"
      
      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      let leng = 0.0;
      if (height > width) {
        leng = height.toFixed(2);
      } else {
        leng = width.toFixed(2);
      };
      let info = prediction.class + ' ' + leng
      const textWidth = ctx.measureText(info).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      let leng = 0.0;
      if (height > width) {
        leng = height.toFixed(2);
      } else {
        leng = width.toFixed(2);
      };
      ctx.fillText(prediction.class + ' ' + leng, x, y);
    });
    // ###

    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}
//window.onload=function(e){try{app()}catch(e){}};
app();

