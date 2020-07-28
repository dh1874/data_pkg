
# uwsgi --ini uwsgi.ini  启动
# uwsgi --stop uwsgi/uwsgi.pid  停止
# uwsgi --reload uwsgi/uwsgi.pid  重启


# 1) 解压
tar -zxvf uWSGI-2.0.19.1.tar.gz -C pkg

# 2）安装
/root/anaconda3/bin/python setup.py install

# 3）环境变量
LD_LIBRARY_PATH=/root/anaconda3/lib:$LD_LIBRARY_PATH
source /root/.bash_profile

# 4) 配置文件
uwsgi.ini  -- 创建&配置
uwsgi.pid  -- 创建
uwsgi.status  -- 创建
