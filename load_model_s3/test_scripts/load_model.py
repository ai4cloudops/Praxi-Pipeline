import boto3
import os
import time
# time.sleep(5000)
vw_model_localpath = '/home/ubuntu/Praxi-Pipeline/load_model_s3/test_scripts/cwd/vw_model.vw'
s3 = boto3.resource(service_name='s3', 
                    region_name='us-east-1', 
                    aws_access_key_id="AKIAXECNQISLO5332P6S", 
                    aws_secret_access_key="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",)
s3.Bucket('praxi-model-1').download_file(Key='praxi-model.vw', Filename=vw_model_localpath)
# os.popen('cp {0} {1}'.format(vw_model_localpath, vw_model_path))

clf_localpath = '/home/ubuntu/Praxi-Pipeline/load_model_s3/test_scripts/cwd/clf.p'
s3.Bucket('praxi-model-1').download_file(Key='praxi-model.p', Filename=clf_localpath)
# os.popen('cp {0} {1}'.format(clf_localpath, clf_path))