import sagemaker
from sagemaker.sklearn.estimator import SKLearn

sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::<account-id>:role/<sagemaker-execution-role>"

sklearn_estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    py_version="py3",
    script_mode=True,
    hyperparameters={"sagemaker_program": "train.py"},
    sagemaker_session=sagemaker_session
)

sklearn_estimator.fit()
