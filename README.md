Disk Failure Prediction

1.Background and instructions:  
    Various disk failures are not rare in large-scale IDCs and cloud computingenvironments, fortunately, we have S.M.A.R.T. (Self-Monitoring, Analysis, andReporting Technology; often written as SMART) logs collected from computer hard disk drives (HDDs), solid-state drives (SSDs) and eMMC drives that detects and reports on various indicators of drive reliability, with the intent of enabling the anticipation of hardware failures.
Since 2013, Backblaze has published statistics and insights based on the hard drivesin their data center, as well as the data underlying these reports. In this case study, you can download SMART logs from Backblaze website(https://www.backblaze.com/b2/hard-drive-test-data.html), then design and implement a machine learning based solution to predict the disk failures in daily(output prediction results in each testing day) granularity. The output document should include detailed illustration of the following parts:  
·The methods or flow of data preprocessing and feature engineering.  
·How to choose machine learning models and tune the parameters?  
·How to evaluate the results?  
·What insights or lessons learned from this task?   
 
2.Inspirations:    
·You may learn more background information from: https://www.backblaze.com/blog/hard-drive-smart-stats/  
·You can consider to evaluate the results in weekly or monthly period.  
·Codes, charts and tables are essential for results interpretation.  
 
3.Datasets and meta description:  
·You can download SMART logs from:https://www.backblaze.com/b2/hard-drive-test-data.html  
·https://en.wikipedia.org/wiki/S.M.A.R.T.  

4.Reference:  
Proactive Prediction of Hard Disk Drive Failure  

Now the folder structure under data folder is sth. like:  
.    
├─data  
│  ├─data_Q1_2018  
│  ├─data_Q2_2018  
│  ├─data_Q3_2018  
│  └─data_Q4_2018  
├─data_preprocess  
├─dataset.csv  
└─KNN.py  
