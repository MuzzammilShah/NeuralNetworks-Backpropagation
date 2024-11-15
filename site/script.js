// JavaScript for Tabs
const tabs = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    tabContents.forEach(c => c.classList.remove('active'));

    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

// Loss Graph Code (as before)
const learningRateInput = document.getElementById("learningRate");
const epochsInput = document.getElementById("epochs");
const trainButton = document.getElementById("trainButton");
const lossChart = document.getElementById("lossChart");
const ctx = lossChart.getContext("2d");

learningRateInput.addEventListener("input", () => {
  document.getElementById("learningRateValue").textContent = learningRateInput.value;
});

epochsInput.addEventListener("input", () => {
  document.getElementById("epochsValue").textContent = epochsInput.value;
});

trainButton.addEventListener("click", () => {
  const losses = simulateTraining(); // Dummy loss simulation
  ctx.clearRect(0, 0, lossChart.width, lossChart.height);
  ctx.beginPath();
  ctx.moveTo(0, lossChart.height);

  losses.forEach((loss, i) => {
    const x = (i / losses.length) * lossChart.width;
    const y = lossChart.height - loss * 100;
    ctx.lineTo(x, y);
  });

  ctx.strokeStyle = "#6200ea";
  ctx.lineWidth = 2;
  ctx.stroke();
});

// Dummy Training Simulation
function simulateTraining() {
  return Array.from({ length: parseInt(epochsInput.value) }, (_, i) => Math.exp(-i * 0.1));
}

// MLP Visualization Code
const mlpCanvas = document.getElementById("mlpCanvas");
const generateMlpButton = document.getElementById("generateMlp");
const trainMlpButton = document.getElementById("trainMlp");

let mlpGraph = []; // Store MLP layers and connections
let learningRate = 0.1;

// Generate MLP Based on User Input
generateMlpButton.addEventListener("click", () => {
  const layers = parseInt(document.getElementById("layers").value);
  const neuronsPerLayer = document.getElementById("neurons").value
    .split(',')
    .map(n => parseInt(n));
  learningRate = parseFloat(document.getElementById("learningRateMlp").value);

  if (layers !== neuronsPerLayer.length) {
    alert("Number of layers must match neurons-per-layer input!");
    return;
  }

  generateMLPGraph(layers, neuronsPerLayer);
});

// Generate and Render MLP Graph
function generateMLPGraph(layers, neuronsPerLayer) {
  mlpGraph = [];
  mlpCanvas.innerHTML = ""; // Clear previous graph

  // Create Nodes and Connections
  for (let layer = 0; layer < layers; layer++) {
    const layerNeurons = [];

    for (let neuron = 0; neuron < neuronsPerLayer[layer]; neuron++) {
      const node = {
        id: `L${layer}N${neuron}`,
        layer,
        neuron,
        value: Math.random().toFixed(2), // Random initial value
        bias: Math.random().toFixed(2),
      };
      layerNeurons.push(node);
    }

    mlpGraph.push(layerNeurons);
  }

  renderMLPGraph();
}

// Render MLP Graph
function renderMLPGraph() {
  mlpCanvas.innerHTML = ""; // Clear the canvas

  mlpGraph.forEach((layer, layerIndex) => {
    const layerDiv = document.createElement("div");
    layerDiv.className = "layer";

    layer.forEach(neuron => {
      const neuronDiv = document.createElement("div");
      neuronDiv.className = "neuron";
      neuronDiv.textContent = `Value: ${neuron.value}\nBias: ${neuron.bias}`;
      layerDiv.appendChild(neuronDiv);
    });

    mlpCanvas.appendChild(layerDiv);
  });

  drawConnections();
}

// Draw Connections Between Layers
function drawConnections() {
  const canvasBounds = mlpCanvas.getBoundingClientRect();

  const connections = document.createElement("svg");
  connections.style.position = "absolute";
  connections.style.width = `${canvasBounds.width}px`;
  connections.style.height = `${canvasBounds.height}px`;

  mlpGraph.forEach((layer, layerIndex) => {
    if (layerIndex === mlpGraph.length - 1) return; // Skip last layer

    const nextLayer = mlpGraph[layerIndex + 1];

    layer.forEach(fromNeuron => {
      nextLayer.forEach(toNeuron => {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", fromNeuron.x);
        line.setAttribute("y1", fromNeuron.y);
        line.setAttribute("x2", toNeuron.x);
        line.setAttribute("y2", toNeuron.y);
        line.setAttribute("stroke", "#6200ea");
        line.setAttribute("stroke-width", "2");
        connections.appendChild(line);
      });
    });
  });

  mlpCanvas.appendChild(connections);
}

// Train MLP
trainMlpButton.addEventListener("click", () => {
  // Example Backpropagation Simulation
  mlpGraph.forEach(layer => {
    layer.forEach(neuron => {
      neuron.value = (parseFloat(neuron.value) - learningRate).toFixed(2);
    });
  });

  renderMLPGraph();
});