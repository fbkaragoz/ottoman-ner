data = readtable('ner_evaluation_report.csv');

% Extract relevant data
entityTypes = data.Properties.RowNames;
precision = data.Precision;
recall = data.Recall;
f1score = data.('F1-Score');
support = data.Support;

% Convert entityTypes to a cell array for the plots
entityTypes = cellstr(entityTypes);

% Precision Bar Plot
figure;
bar(precision);
set(gca, 'xticklabel', entityTypes);
title('Precision for Each Entity Type');
ylabel('Precision');
xlabel('Entity Type');
colormap('winter');

% Recall Bar Plot
figure;
bar(recall);
set(gca, 'xticklabel', entityTypes);
title('Recall for Each Entity Type');
ylabel('Recall');
xlabel('Entity Type');
colormap('summer');

% F1-Score Bar Plot
figure;
bar(f1score);
set(gca, 'xticklabel', entityTypes);
title('F1-Score for Each Entity Type');
ylabel('F1-Score');
xlabel('Entity Type');
colormap('autumn');

% Support Bar Plot
figure;
bar(support);
set(gca, 'xticklabel', entityTypes);
title('Support for Each Entity Type');
ylabel('Support');
xlabel('Entity Type');
colormap('spring');