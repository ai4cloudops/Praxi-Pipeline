def cp_tagsets():
    import os
    packages_l = []
    packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1"]
    # packages_l_0 = ["s3fs", "yarl", "psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli", "pyyaml", "rsa", "s3transfer", "urllib3", "setuptools", "typing-extensions", "charset-normalizer", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata", "emoji", "Flask", "seaborn", "NLTK", "pytest", "zipp", "authlib", "pycparser", "colorama", "oauthlib", "pandas", "pillow", "matplotlib", "scipy", "boto3", "cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", ]
    # packages_l_1 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh", "dask", "deap", "pyspark", "nilearn"]
    # packages_l_2 = ["networkx", "SQLAlchemy", "scikit-image", "scoop", "Theano", "beautifulsoup4", "Scrapy", "plotly", "pycaret","mahotas","statsmodels"]
    # packages_l.extend(packages_l_0)
    # packages_l.extend(packages_l_1)
    # packages_l.extend(packages_l_2)
    from itertools import combinations
    # for length in range(1, len(packages_l)+1):
    count = 0
    for length in range(2,3):  # choose `length` amount of packages
        for package_names in combinations(packages_l, length):
            # print(package_names)
            dirname = os.getcwd()
            # dirname = ''
            # dirname += "/loaded_data"1
            dirname = '/home/ubuntu/Praxi-Pipeline/data/data'
            # out_dirname = dirname
            target_dir_train = "/home/ubuntu/Praxi-Pipeline/data/post_train_ml"
            target_dir_test = "/home/ubuntu/Praxi-Pipeline/data/post_test"
            out_dirname = dirname+"/data/"+"-".join(package_names)+'-'+"tagsets/"
            # print(out_dirname)
            # print(out_dirname)
            # print(os.path.exists(out_dirname))
            if (os.path.exists(out_dirname)):
                count += 1
                spec_count = 0
                tagsets_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name)]
                # print(tagsets_l)
                # print(tagsets_l)
                # if len(tagsets_l) == 2:
                for tagsets_name in tagsets_l:
                    # print(out_dirname+tagsets_name, target_dir+"/" +tagsets_name)
                    if (os.path.exists(out_dirname+tagsets_name)) and (spec_count < 2):
                        # print('1',tagsets_name)
                        os.popen('cp {0} {1}'.format(out_dirname+tagsets_name, target_dir_test +"/"+tagsets_name))
                        # shutil.copy2(out_dirname+tagsets_name, '/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/'))
                    elif (os.path.exists(out_dirname+tagsets_name)) and (spec_count < 32):
                        # print('2',tagsets_name)
                        os.popen('cp {0} {1}'.format(out_dirname+tagsets_name, target_dir_train +"/"+tagsets_name))
                    spec_count +=1
                print(spec_count)
                

    print(count)

if __name__ == "__main__":
    cp_tagsets()