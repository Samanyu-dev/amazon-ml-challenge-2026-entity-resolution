import sagemaker, boto3, sys
from sagemaker.pytorch import PyTorch
sess = sagemaker.Session(boto_session=boto3.Session(region_name='ap-south-1'), default_bucket='er-ml26-908404959957-aps1')
est = PyTorch(entry_point='sm_llm_entry.py', source_dir='/private/tmp/claude-501/sm_llm_src', framework_version='2.3', py_version='py311',
              role='arn:aws:iam::908404959957:role/AmazonSageMaker-ExecutionRole-ml', instance_count=1, instance_type='ml.g5.2xlarge',
              volume_size=150, max_run=12 * 3600, sagemaker_session=sess, base_job_name='er-llm7b',
              checkpoint_s3_uri='s3://er-ml26-908404959957-aps1/checkpoints/llm7b_v2', checkpoint_local_path='/opt/ml/checkpoints',
              output_path='s3://er-ml26-908404959957-aps1/output', disable_profiler=True)
try:
    est.fit({'pairs': 's3://er-ml26-908404959957-aps1/pairs_llm/'}, wait=False); print('LAUNCHED', est.latest_training_job.name)
except Exception as e:
    print('NOT_YET', type(e).__name__, str(e)[:200]); sys.exit(3)
