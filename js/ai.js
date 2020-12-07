
function AI() {
    this.num_inputs = 16;
    this.num_actions = 4;
    this.num_games = 10;
    this.brain = new deepqlearn.Brain(this.num_inputs, this.num_actions, this.getOpt());
    // this.setBrainWindow();
    this.fileName = "network.json"
    this.trainedNet = false;
    // this.gameManager = gameManager;
}

// AI.prototype.createNet = function () {
//     this.brain = new deepqlearn.Brain(this.num_inputs, this.num_actions, this.getOpt());
// }


AI.prototype.trainAi = async function (gameManager) {
    // console.log(this.trainedNet);
    let timeout = animationDelay;
    let count = 0;
    
    for (let i = 0; i < this.num_games; i++) {
        await this.trainGame(gameManager);
        this.setBrainWindow();
        count += 1;
        console.log(count);
    }

}

AI.prototype.trainGame = async function (gameManager) {
    let count = 0;
    gameManager.restart();
    while (!gameManager.isGameTerminated()) {
        let prevScore = gameManager.score;
        let prevInputObj = this.getInputObj(gameManager.grid.cells);
        let action = this.brain.forward(prevInputObj.inputArray);
        gameManager.moveTrain(action);
        let currScore = gameManager.score;
        let currInputObj = this.getInputObj(gameManager.grid.cells);
        let reward = this.getReward(gameManager, prevInputObj, currInputObj);
        if (currScore > prevScore) reward += currScore-prevScore;
        // console.log(reward);
        // if (currScore > prevScore) reward += 10;
        
        // if (currInputObj.count < prevInputObj.count) reward += 5;
        // if (currInputObj.count == prevInputObj.count && prevInputObj.total == currInputObj.total) reward -= 1;
        // if (gameManager.over && !gameManager.won) reward -= 30;
        // if (gameManager.won) reward += 100; 
        this.brain.backward(reward);
        if (gameManager.won) gameManager.keepPlaying = false;
        // console.log(action);
        // console.log(reward);
        // setTimeout(this.setBrainWindow, timeout);
        // this.setBrainWindow();
    }
}

AI.prototype.getReward = function (gameManager, prevInputObj, currInputObj) {
    let reward = 0;
    
    if (currInputObj.count < prevInputObj.count) reward += prevInputObj.count-currInputObj.count;
    if (currInputObj.count == prevInputObj.count && prevInputObj.total == currInputObj.total) reward -= 1;
    if (gameManager.over && !gameManager.won) reward -= 500;
    if (gameManager.won) reward += 1000; 
    if (currInputObj.inputArray[0] == currInputObj.highestValue) reward += 10;
    for (let index of currInputObj.weightedSquares) {
        if (currInputObj.inputArray[index] == currInputObj.highestValue) reward += 5;
        else if (currInputObj.inputArray[index] == currInputObj.highestValue/2) reward += 5;
        else if (currInputObj.inputArray[index] == currInputObj.highestValue/4 ) reward += 5;
        else if (currInputObj.inputArray[index] == currInputObj.highestValue/8 ) reward += 5;
    }

    return reward;
}


AI.prototype.getMove = function (grid) {
    let inputArray = this.getInputObj(grid).inputArray;
    let action = this.brain.forward(inputArray);
    return action;
}





// Unchanged options from the demo page of ConvNetJS
AI.prototype.getOpt = function () {
    let temporal_window = 1; // amount of temporal memory. 0 = agent lives in-the-moment :)
    let network_size = this.num_inputs*temporal_window + this.num_actions*temporal_window + this.num_inputs;

    // the value function network computes a value of taking any of the possible actions
    // given an input state. Here we specify one explicitly the hard way
    // but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
    // to just insert simple relu hidden layers.
    let layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});


    // layer_defs.push({type:'fc', num_neurons: 4096, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 2048, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 1024, activation:'relu'});




    // Neural Network 2
    layer_defs.push({type:'fc', num_neurons: 512, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 256, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 128, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons: 128, activation:'relu'});
    
    
    
    // Neural Network 1
    // layer_defs.push({type:'fc', num_neurons: 64, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 32, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 16, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 8, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 8, activation:'relu'});



    // Neural Network 3
    // layer_defs.push({type:'fc', num_neurons: 16, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 8, activation:'relu'});
    // layer_defs.push({type:'fc', num_neurons: 8, activation:'relu'});




    layer_defs.push({type:'regression', num_neurons:this.num_actions});

  



    // Options taken from https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html

    let tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

    let opt = {};
    opt.temporal_window = temporal_window;
    opt.experience_size = 30000;

    opt.gamma = 0.7;

    opt.start_learn_threshold = 1000;
    opt.learning_steps_total = 200000;
    opt.learning_steps_burnin = 3000;
    opt.epsilon_test_time = 0.05;



    // opt.learning_steps_total = 0;
    // opt.learning_steps_burnin = 0;
    // opt.start_learn_threshold = 0;
    // opt.epsilon_test_time = 0;
    
    
    opt.epsilon_min = 0.05;
    
    opt.layer_defs = layer_defs;
    opt.tdtrainer_options = tdtrainer_options;
    return opt;
}


AI.prototype.getInputObj = function (grid) {
    // console.dir(grid);
    let inputObj = {}
    let inputArray = new Array(16);
    let index = -1;
    let count = 0;
    let highestValue = -1;
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            if (grid[i][j]) {
                if (grid[i][j].value > highestValue) highestValue = grid[i][j].value;
                inputArray[++index] = grid[i][j].value;
                count += 1;
            } 
            else inputArray[++index] = 0;
        }
    }
    inputObj.inputArray = inputArray;
    inputObj.count = count;
    inputObj.total = inputArray.reduce((x, y) => { return x+y }, 0);
    inputObj.highestValue = highestValue;
    inputObj.weightedSquares = [4, 8, 12];

    // console.log(inputObj.total);
    return inputObj;
}



AI.prototype.saveNet = function () {
    let json = this.brain.value_net.toJSON();
    json = JSON.stringify(json);
    let a = document.createElement('a');
    a.href = "data:application/octet-stream,"+encodeURIComponent(json);
    a.download = this.fileName;
    a.click();
}


AI.prototype.loadNet = async function () {
    loadFile(this);
    this.trainedNet = true;
    // this.stopLearning();
}


AI.prototype.setBrainWindow = function () {
    let element = document.getElementById("brain-info");
    this.brain.visSelf(element);
}


AI.prototype.stopLearning = function () {
    this.brain.learning = false;
}












async function loadFile(ai) {
    let fileInput = document.getElementById("input-net-file");
    let files = fileInput.files;
    if (files.length == 0) return;

    let file = files[0];

    let fileReader = new FileReader();

    fileReader.onerror = function (e) {
        alert("Error!");
        return;
    }
    
    fileReader.readAsText(file);

    fileReader.onloadend = function (event) {
        file = fileReader.result;
        let json = JSON.parse(file);
        try {

            ai.brain.value_net.fromJSON(json);
            let p = document.getElementById("net-loaded-textarea");
            p.innerHTML = "Neural Network Loaded";
            // fileInput.value = "";
        }
        catch (error) {
            alert (error);
        }
    }

}