function likelihood = getLikelihood(feature,y,fValue,target)
  %get the likelihood of feature value(fValue) in feature and target class
  %from dataset
  
  targetRows = find(y==target);
  useFeature = feature(targetRows);
  
  varFeature = var(useFeature);
  meanFeature = mean(useFeature);
  
  likelihood = (1/sqrt(2*pi*varFeature))...
                  *exp((-1*(fValue-meanFeature)^2)/(2*varFeature));
              
end