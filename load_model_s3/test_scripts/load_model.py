import boto3
import os
import time
# time.sleep(5000)

s3 = boto3.resource(service_name='s3', 
                    region_name='us-east-1', 
                    aws_access_key_id="AKIAXECNQISLIBBXAMLV", 
                    aws_secret_access_key="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",)
# vw_model_localpath = '/home/ubuntu/Praxi-Pipeline/load_model_s3/test_scripts/cwd/vw_model.vw'
# s3.Bucket('praxi-model-1').download_file(Key='praxi-model.vw', Filename=vw_model_localpath)
# # os.popen('cp {0} {1}'.format(vw_model_localpath, vw_model_path))

# clf_localpath = '/home/ubuntu/Praxi-Pipeline/load_model_s3/test_scripts/cwd/clf.p'
# s3.Bucket('praxi-model-1').download_file(Key='praxi-model.p', Filename=clf_localpath)
# # os.popen('cp {0} {1}'.format(clf_localpath, clf_path))

xgb_model_localpath = '/home/ubuntu/Praxi-Pipeline/load_model_s3/test_scripts/cwd/model.json'
s3.Bucket('praxi-model-xgb-02').download_file(Key='model.json', Filename=xgb_model_localpath)


# s3.Bucket('praxi-interm-1').upload_file('/home/ubuntu/Praxi-Pipeline/load_model_s3/test_scripts/cwd/clf.p', "clf.p")