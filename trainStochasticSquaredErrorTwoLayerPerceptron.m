function [hiddenWeights, outputWeights, error] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    fixrandomWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    fixrandomWeights = fixrandomWeights./size(fixrandomWeights, 2);
    
    n = zeros(1,batchSize);
    
    %记录遍历过的训练
    total = zeros(size(inputValues,2),1);
    
    figure; hold on;
    
    times = 1;
    
    for iter = 1: times
    TrSeqOrd = randperm(trainingSetSize);
    for t = 1: epochs
        for k = 1: batchSize
            % Select which input vector to train on.
            % n(k) = floor(rand(1)*trainingSetSize + 1);
            n(k) = TrSeqOrd((t-1) * batchSize + k);
            total(n(k)) = 1;
            % Propagate the input vector through the network.
            inputVector = inputValues(:, n(k));%输入值
            hiddenActualInput = hiddenWeights*inputVector;%隐藏层权重和,矩阵700*1
            hiddenOutputVector = activationFunction(hiddenActualInput);%隐藏层的激活值，矩阵700*1
            outputActualInput = outputWeights*hiddenOutputVector;%输出层的权重和
            outputVector = activationFunction(outputActualInput);%输出层的激活值，矩阵10*1
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);%矩阵10*1
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
            %hiddenDelta = dActivationFunction(hiddenActualInput).*(fixrandomWeights'*outputDelta);
            
            outputWeights = outputWeights - learningRate.*outputDelta*hiddenOutputVector';
            hiddenWeights = hiddenWeights - learningRate.*hiddenDelta*inputVector';
        end;
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2);%求范数
        end;
        error = error/batchSize;
        
        plot(t, error,'*');
    end;
    end;
    save('interVariable.mat','total');
end