import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
import time

if __name__ == "__main__":
    
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sagemaker_session.default_bucket()
    
    input_data_param = ParameterString(
        name="InputData", 
        default_value=f"s3://{bucket}/raw_data/"
    )
    
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type="ml.t3.medium",
        instance_count=1,
        base_job_name="devops-preprocess",
        role=role,
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        code="preprocess.py",
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=input_data_param,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            sagemaker.processing.ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            sagemaker.processing.ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
    )

    sklearn_estimator = SKLearn(
        entry_point="train.py",
        framework_version="1.0-1",
        instance_type="ml.t3.medium",
        role=role,
        hyperparameters={"max_iter": 1000},
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=sklearn_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            )
        },
    )

    evaluation_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type="ml.m4.xlarge",
        instance_count=1,
        base_job_name="devops-evaluate",
        role=role,
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        code="evaluate.py",
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            sagemaker.processing.ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ],
    )

    pipeline_name = f"DevOpsEffectivenessPipeline-{int(time.time())}"
    
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_data_param],
        steps=[step_preprocess, step_train, step_evaluate],
    )

    print(f"Upserting pipeline: {pipeline_name}")
    pipeline.upsert(role_arn=role)

    print(f"Starting pipeline execution...")
    execution = pipeline.start()
    print(f"Pipeline started. Execution ARN: {execution.arn}")