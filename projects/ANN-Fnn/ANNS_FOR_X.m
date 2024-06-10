%  Õ„Ì· «·»Ì«‰« 
[alphabets, targets] = prprob;
%  ÂÌ∆… «·„ €Ì—« 
inputSize = size(alphabets, 1);
hiddenSize = 20; % “Ì«œ… ⁄œœ «·⁄’»Ê‰«  ›Ì «·ÿ»ﬁ… «·Œ›Ì…
outputSize = size(targets, 1);
learningRate = 0.005; %  ﬁ·Ì· „⁄œ· «· ⁄·„
%  ÂÌ∆… «·Ê“‰ Ê«· ÕÌ“
W1 = rand(hiddenSize, inputSize) * 0.1 - 0.05;
b1 = rand(hiddenSize, 1) * 0.1 - 0.05;
W2 = rand(outputSize, hiddenSize) * 0.1 - 0.05;
b2 = rand(outputSize, 1) * 0.1 - 0.05;
% œ«·… «· ÕÊÌ·
sigmoid = @(x) 1 ./ (1 + exp(-x));
% «· œ—Ì»
for epoch = 1:500
    %  ﬁ”Ì„ «·»Ì«‰«  ·· Õﬁﬁ «·„ ﬁ«ÿ⁄
    cv = cvpartition(size(alphabets, 2), 'KFold', 5);
    for i = 1:cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        % «· œ—Ì» »«” Œœ«„ »Ì«‰«  «· œ—Ì» ›ﬁÿ
        for c = find(trainIdx)'
            % «·≈‘«—… «·√„«„Ì… 
            a1 = alphabets(:, c);
            z2 = W1 * a1 + b1;
            a2 = sigmoid(z2);
            z3 = W2 * a2 + b2;
            a3 = sigmoid(z3);
            % «·Œÿ√
            delta3 = a3 - targets(:, c);
            delta2 = (W2' * delta3) .* a2 .* (1 - a2);
            %  ÕœÌÀ «·Ê“‰ Ê«· ÕÌ“
            W2 = W2 - learningRate * delta3 * a2';
            b2 = b2 - learningRate * delta3;
            W1 = W1 - learningRate * delta2 * a1';
            b1 = b1 - learningRate * delta2;
        end
        %  ﬁÌÌ„ «·√œ«¡ »«” Œœ«„ »Ì«‰«  «·«Œ »«—
        for c = find(testIdx)'
            % «·≈‘«—… «·√„«„Ì…
            a1 = alphabets(:, c);
            z2 = W1 * a1 + b1;
            a2 = sigmoid(z2);
            z3 = W2 * a2 + b2;
            a3 = sigmoid(z3)  
        end
    end
end
 
% «Œ »«— «·‘»ﬂ…
c = alphabets(:,3);
output = sigmoid(W2 * sigmoid(W1 * c + b1) + b2);
[~, answer] = max(output);
figure; plotchar(alphabets(:, answer));
% «Œ »«— «·‘»ﬂ… „⁄ c «·„‘Ê‘
noisyg = c + randn(35,1) * 0.2;
figure; plotchar(noisyg);
output2 = sigmoid(W2 * sigmoid(W1 * noisyg + b1) + b2);
[~, answer2] = max(output2);
figure; plotchar(alphabets(:, answer2));
 
