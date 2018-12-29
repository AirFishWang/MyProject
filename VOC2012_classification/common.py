# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     common
   Description :
   Author :        wangchun
   date：          18-12-21
-------------------------------------------------
   Change Activity:
                   18-12-21:
-------------------------------------------------
"""
import os
import logging
import logging.handlers

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)

# 创建一个handler，用于写入日志文件
log_name = "voc2012.log"
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# fh = logging.FileHandler(os.path.join(log_dir, log_name))
fh = logging.handlers.TimedRotatingFileHandler(os.path.join(log_dir, log_name), 'midnight', 1, 365)
# fh.setLevel(logging.INFO)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)