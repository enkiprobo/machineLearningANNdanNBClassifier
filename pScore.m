function score = pScore(actuals, predicts, option)
  %this function will return the performance score
  %option = 1, will use F1-micro average
  %option = 2, will use F1-macro average
  %option = 3, will use simple accuracy
  
  %make confussion matrix
  nClass = max(actuals);
  cM = zeros(nClass);
  for i=1:length(actuals(:,1))
    cM(actuals(i), predicts(i)) = cM(actuals(i), predicts(i)) + 1;
  end
  
  %initiate the measure tools
  TP = zeros(nClass,1);
  FP = zeros(nClass,1);
  FN = zeros(nClass,1);
  TN = zeros(nClass,1);
  for i=1:nClass,
    TP(i) = cM(i,i);
    FP(i) = sum(cM(:,i))-TP(i);
    FN(i) = sum(cM(i,:))-TP(i);
    TN(i) = sum(sum(cM))-sum(FP)-sum(FN)-TP(i);
  end
  
  %calculate the performance score based on the selected option
  if(option==1),
 
     P = sum(TP)/(sum(TP)+sum(FP));
     R = sum(TP)/(sum(TP)+sum(FN));
     
     score = 2*P*R/(P+R);
  elseif(option==2),
  
    f1List = zeros(nClass,1);
    for i=1:nClass,
      P = TP(i)/(TP(i)+FP(i));
      R = TP(i)/(TP(i)+FN(i));
       
      f1List(i) = 2*P*R/(P+R);
    end
    
    score = sum(f1List)/nClass;
  elseif(option==3),
    
    sumRight = 0;
    for i=1:nClass,
      sumRight = sumRight+TP(i);
    end
    
    score = sumRight/sum(sum(cM));
  end
  
  %make the score in percent
  score = score*100;
end