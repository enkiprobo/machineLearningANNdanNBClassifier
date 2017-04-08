function [X y] = loadDatasets(filename),
    %this function will load the dataset and will 
    %divide it to 2 variable. X as data and y as label

    dataset = csvread(filename);
    X = dataset(:,1:2);
    y = dataset(:,3);
end