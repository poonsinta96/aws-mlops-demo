# AWS MLOps Demonstration with SageMaker

This demonstration shows how AWS SageMaker enables MLOps by using SageMaker Experiment, SageMaker Pipelines and SageMaker Model Monitor.

Important: this application uses various AWS services and there are costs associated with these services - please see the [AWS Pricing page](https://aws.amazon.com/pricing/) for details. You are responsible for any AWS costs incurred. No warranty is implied in this example.

```bash
.
├── README.MD                   <-- This instructions file
├── ds-experiment-demo          <-- Data Science Component - includes the data and the main python notebook which creates the model
├── mlops-pipeline-demo         <-- Machine Learning Pipeline Component - Source code for SageMaker Pipeline and SageMaker Project template 
├── monitoring-demo             <-- Model Monitoring Components - python notebook which sets up Model Monitoring and feeds data to it
├── img                         <-- image files for the README.md

```

## Demonstration Explanation 

The following image is the architecture diagram of the entire project.

![alt text](https://github.com/poonsinta96/aws-mlops-demo/blob/main/img/MLOps%20Architecture%20Diagram.png?raw=true)

It can be further broken down into four main segments as shown below.

![alt text](https://github.com/poonsinta96/aws-mlops-demo/blob/main/img/MLOps%20Architecture%20Diagram%20Explained.png?raw=true)

### Experimentation Stage

The Experimentation Stage is meant to show the typical workflow of a data scientist when they are selecting, training and testing their model.  

Architecture Diagram Component(s): Experimentation

The source code for the experimentation stage can be found in _ds-experiment-demo_ folder. In our project, we only retained the essential components however, the full version of this code can also be found in the following link: https://aws.amazon.com/getting-started/hands-on/build-train-deploy-monitor-machine-learning-model-sagemaker-studio/?trk=gs_card 

### ML Pipeline Stage

The ML Pipeline Stage is meant to show how SageMaker provide an end-to-end MLOps solution from the creation of pipeline to the deployment of end-point in production. 

Architecture Diagram Component(s): Refactoring and CI/CD

The source code for the ML Pipeline Stage can be found in _mlops-pipeline-demo_ however, to successfully run this stage, you have to create a SageMaker Project from within SageMaker Studio and manually import certain folders and edit certain files. Below will be the steps required to run this stage.

1. Create SageMaker Project
2. Change the CodeBuild Component
3. Change the CodeDeploy Component 


