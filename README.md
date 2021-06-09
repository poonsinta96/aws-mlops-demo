# AWS MLOps Demonstration with SageMaker

This demonstration shows how AWS SageMaker enables MLOps by using SageMaker Experiment, SageMaker Pipelines and SageMaker Model Monitor.

Important: this application uses various AWS services and there are costs associated with these services after the Free Tier usage - please see the [AWS Pricing page](https://aws.amazon.com/pricing/) for details. You are responsible for any AWS costs incurred. No warranty is implied in this example.

```bash
.
├── README.MD                   <-- This instructions file
├── ds-experiment-demo          <-- Data Science Component - includes the data and the main python notebook which creates the model
├── mlops-pipeline-demo         <-- Machine Learning Pipeline Component - Source code for SageMaker Pipeline and SageMaker Project template 
├── monitoring-demo             <-- Model Monitoring Components - python notebook which sets up Model Monitoring and feeds data to it
```

## Explanation 

![alt text](https://github.com/poonsinta96/aws-mlops-demo/blob/main/img/MLOps%20Architecture%20Diagram.png?raw=true)
