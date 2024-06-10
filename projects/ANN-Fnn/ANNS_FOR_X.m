% ����� ��������
[alphabets, targets] = prprob;
% ����� ���������
inputSize = size(alphabets, 1);
hiddenSize = 20; % ����� ��� ��������� �� ������ ������
outputSize = size(targets, 1);
learningRate = 0.005; % ����� ���� ������
% ����� ����� �������
W1 = rand(hiddenSize, inputSize) * 0.1 - 0.05;
b1 = rand(hiddenSize, 1) * 0.1 - 0.05;
W2 = rand(outputSize, hiddenSize) * 0.1 - 0.05;
b2 = rand(outputSize, 1) * 0.1 - 0.05;
% ���� �������
sigmoid = @(x) 1 ./ (1 + exp(-x));
% �������
for epoch = 1:500
    % ����� �������� ������ ��������
    cv = cvpartition(size(alphabets, 2), 'KFold', 5);
    for i = 1:cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        % ������� �������� ������ ������� ���
        for c = find(trainIdx)'
            % ������� �������� 
            a1 = alphabets(:, c);
            z2 = W1 * a1 + b1;
            a2 = sigmoid(z2);
            z3 = W2 * a2 + b2;
            a3 = sigmoid(z3);
            % �����
            delta3 = a3 - targets(:, c);
            delta2 = (W2' * delta3) .* a2 .* (1 - a2);
            % ����� ����� �������
            W2 = W2 - learningRate * delta3 * a2';
            b2 = b2 - learningRate * delta3;
            W1 = W1 - learningRate * delta2 * a1';
            b1 = b1 - learningRate * delta2;
        end
        % ����� ������ �������� ������ ��������
        for c = find(testIdx)'
            % ������� ��������
            a1 = alphabets(:, c);
            z2 = W1 * a1 + b1;
            a2 = sigmoid(z2);
            z3 = W2 * a2 + b2;
            a3 = sigmoid(z3)  
        end
    end
end
 
% ������ ������
c = alphabets(:,3);
output = sigmoid(W2 * sigmoid(W1 * c + b1) + b2);
[~, answer] = max(output);
figure; plotchar(alphabets(:, answer));
% ������ ������ �� c ������
noisyg = c + randn(35,1) * 0.2;
figure; plotchar(noisyg);
output2 = sigmoid(W2 * sigmoid(W1 * noisyg + b1) + b2);
[~, answer2] = max(output2);
figure; plotchar(alphabets(:, answer2));
 
