#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-04 22:52:37
# @Author  : HD

import numpy as np
import matplotlib as mpl
import sys
# from scipy.misc import imread
from scipy.interpolate import spline
from configuration import *
if IS_LINUX:
    print('[plotHD] initial fig_path = %s ' % FIGURE_FILE_PHAT)
    print('[plotHD] setEnv = LINUX')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_line_space(x_list, y_list, point_nums):
    """
    根据输入的一组X, Y数组，生成一组新的X, Y数组，使描点所成的连线，变得平滑
    :param x_list:
    :param y_list:
    :param point_nums:
    :return:
    """
    x_list_new = np.linspace(np.array(x_list).min(),
                             np.array(x_list).max(),
                             point_nums)

    y_list_new = spline(np.array(x_list), np.array(y_list), x_list_new)

    return x_list_new, y_list_new


class MyPlot(object):
    """
    APPLICATION FROM SEABORN
    画图基础操作集成封装类
    """

    """
    风格参数
    """
    __font = 'DejaVu Sans'  # 字体
    __size = 1  # 画布元素大小
    __style = 'whitegrid'  # 风格
    __palette = 'pastel'  # 色彩空间
    __length = '16'  # 长
    __width = '9'  # 宽
    __fig_path = sys.path[0]  # 保存图片默认路径，执行脚本的pwd
    __dpi = 100  # 保存图片的清晰度
    __spline_point_nums = 300

    """
    图片参数
    """
    __x_label = None
    __y_label = None
    __x_tick = None
    __title = None
    __x_lim = None  # [min, max]
    __y_lim = None  # [min, max]
    __is_spline = None  # 是否平滑处理
    __text_dict = {}  # 图片中的标注内容 # dict = {text: [x, y]}

    def __init__(self):
        super(MyPlot, self).__init__()

        self.init_style_params()

    def init_style_params(self):
        """
        初始化风格参数
        :return:
        """
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        sns.set(font=self.__font,
                font_scale=self.__size,
                style=self.__style,
                palette=self.__palette,
                rc={'figure.figsize': (float(self.__length), float(self.__width))})

    def __init_figure_params(self, ax):
        """
        加载图片参数

        :return:
        """

        if self.__x_label:

            plt.xlabel(self.__x_label)

        if self.__y_label:

            plt.ylabel(self.__y_label)

        if self.__title:

            plt.title(self.__title)

        if self.__x_lim:

            plt.xlim(self.__x_lim[0], self.__x_lim[1])  # type() == list()

        if self.__y_lim:

            plt.ylim(self.__y_lim[0], self.__y_lim[1])  # type() == list()

        if self.__x_tick:

            ax.set_xticks(self.__x_tick)

        if len(self.__text_dict) > 0:

            for k, v in self.__text_dict.items():

                ax.text(v[0], v[1], k, fontsize=11.5)

    def __reset_figure_params(self):
        """
        重置图片参数
        :return:
        """

        self.__x_label = None
        self.__y_label = None
        self.__x_tick = None
        self.__title = None
        self.__x_lim = None  # [min, max]
        self.__y_lim = None  # [min, max]
        self.__is_spline = None
        self.__text_dict = {}

    def plot_by_xy(self, x_list, y_list):

        # 初始化，画布
        fig, ax = plt.subplots()

        # 曲线平滑处理
        if self.__is_spline:

            x_list, y_list = make_line_space(x_list, y_list, self.spline_point_nums)

        plt.plot(x_list, y_list)

        ax.yaxis.grid(True)

        # 加载图片参数
        self.__init_figure_params(ax)

        # 展示，或者保存
        self.__show_or_save()

        # 重置图片参数
        self.__reset_figure_params()

    def plot_by_x_group_y(self, x_list, y_dict):
        """
        多组Y， 在同一X轴

        :param x_list:
        :param y_dict: {label: [y]}
        :return:
        """

        fig, ax = plt.subplots()

        for key in y_dict.keys():

            y_list = y_dict[key]

            if self.__is_spline:

                x_list, y_list = make_line_space(x_list, y_list, self.spline_point_nums)

            if key == 'base':

                plt.plot(x_list, y_list, 'r--', label=key)

            else:

                plt.plot(x_list, y_list, label=key)

        plt.legend(loc='best')

        self.__init_figure_params(ax)

        self.__show_or_save()

        self.__reset_figure_params()

    def plot_by_group_x_y(self, x_dict, y_dict):
        """
        多组【X, Y】, 在同一张图. 相同的KEY

        多模型的RUC曲线对比

        :param x_dict：{key: [value]}
        :param y_dict：{key: [value]}
        :return:
        """

        fig, ax = plt.subplots()

        x_keys = x_dict.keys()

        for key in x_keys:

            x_list = x_dict[key]

            y_list = y_dict[key]

            if self.__is_spline:

                x_list, y_list = make_line_space(x_list, y_list, self.spline_point_nums)

            plt.plot(x_list, y_list, label=key)

        plt.legend(loc='best')

        self.__init_figure_params(ax)

        self.__show_or_save()

        self.__reset_figure_params()

    def kde_plot(self, df, col_feature):
        """
        单指标核密度估计
        :param df:
        :param col_feature: str
        :return:
        """
        sns.kdeplot(df[col_feature], shade=True)

        self.__show_or_save()

        self.__reset_figure_params()

    def kde_plot_by_group(self, df, col_feature_name, gropu_col_name, group_values=False):
        """
        多分组，同一指标的 KDE图
        :return:
        """

        raw_grp_value_list = df[[gropu_col_name]].drop_duplicates()[gropu_col_name].tolist()

        grp_value_list = group_values if group_values else raw_grp_value_list

        for value_ in grp_value_list:

            sns.kdeplot(df[df[gropu_col_name] == value_][col_feature_name], shade=True, label=value_)

        self.__show_or_save()

        self.__reset_figure_params()

    def cor_metric_plot(self, df, columns, mask):
        """指定特征两两之间的相关性【未完成】"""

        feature_values = []

        for col in columns:

            feature_values.append(np.array(df[col]))

        cor_metric = np.corrcoef(feature_values)

        cor_metric_df = pd.DataFrame(cor_metric, columns=columns, index=columns)

        sns.heatmap(cor_metric_df, annot=True, mask=mask)

        self.__show_or_save()

        self.__reset_figure_params()

    def lm_point_plot_by_xy(self, df, x_col_name, y_col_name, is_fit_reg):
        """
        :param df:
        :param x_col_name:
        :param y_col_name:
        :param is_fit_reg:
        :return:
        """

        sns.lmplot(x_col_name, y_col_name, df, fit_reg=is_fit_reg)

        self.__show_or_save()

        self.__reset_figure_params()

    def lm_point_plot_by_group_on_x(self, df, x_col_name, y_col_name, group_col_name, is_fit_reg):
        """

        横向展示，多分组下X, Y的散点图例

        :param df:
        :param x_col_name:
        :param y_col_name:
        :param group_col_name: 分组列名
        :param is_fit_reg:
        :return:
        """

        sns.lmplot(x_col_name, y_col_name, df, fit_reg=is_fit_reg, col=group_col_name)

        self.__show_or_save()

        self.__reset_figure_params()

    def lm_point_plot_by_group(self, df, x_col_name, y_col_name, group_col_name, is_fit_reg):
        """

        同张图，多分组下X, Y的散点图例， 不同颜色

        :param df:
        :param x_col_name:
        :param y_col_name:
        :param group_col_name: 分组列名
        :param is_fit_reg:
        :return:
        """

        sns.lmplot(x_col_name, y_col_name, df, fit_reg=is_fit_reg, hue=group_col_name)

        self.__show_or_save()

        self.__reset_figure_params()

    def joint_plot(self, df, x_col_name, y_col_name, kind='scatter'):
        """
        两组数据的分布散点图
        """
        sns.jointplot(x=x_col_name, y=y_col_name, data=df, kind=kind)

        self.__show_or_save()

        self.__reset_figure_params()

    def box_plot(self, df, col_name):
        """
        单变量的箱图

        :param df:
        :param col_name:
        :return:
        """
        sns.boxplot(x=df[col_name])

        self.__show_or_save()

        self.__reset_figure_params()

    def box_plot_by_group(self, df, col_name, grp_col_name):
        """
        分组在指定特征上的箱线图

        :param df:
        :param col_name:
        :param grp_col_name:
        :return:
        """

        f, ax = plt.subplots()

        # Plot the orbital period with horizontal boxes
        sns.boxplot(x=col_name, y=grp_col_name, data=df, whis=np.inf)

        # Add in points to show each observation
        sns.swarmplot(x=col_name, y=grp_col_name, data=df, size=2, color=".3", linewidth=0)

        ax.xaxis.grid(True)

        ax.set(ylabel=grp_col_name)

        sns.despine(trim=True, left=True)

        self.__init_figure_params(ax)

        self.__show_or_save()

        self.__reset_figure_params()

    def bar_plot(self, df, fea_col_name, group_col_name):
        """
        条形图

        :return:
        """

        # 排序，保证条形图，一定程度的单调
        df = df.sort_values(fea_col_name, ascending=False)

        sns.barplot(x=fea_col_name, y=group_col_name, data=df, ci=0.01)

        self.__show_or_save()

        self.__reset_figure_params()

    def bar_plot_by_group(self, df, valueCol, grpCol, params =None, pngName='barPlot'):
        """
        分组条形图

        Parameters
        ----------
        df : DataFrame。 未经聚合操作的宽表对象。 经过聚合操作的宽表对象应该也可以吧。？？【待验证】

            grpCol | valueCol
            ----   |  ----

        valueCol : list or str
            case list => 单分组不同特征的数值对比。【单向二重条形图，总分】
            case str  => 单分组在单特征的数值展示。【普通条形图，纵or横】
            【双向条形图】

        grpCol : str
            分组的列名称

        """

        df = df.sort_values(valueCol[0], ascending=False)

        f, ax = plt.subplots(figsize=(6, 15))

        for value_, palette in zip(valueCol, ['pastel', 'muted']):

            sns.set_color_codes(palette)

            sns.barplot(x=value_, y=grpCol, data=df,
                    label=value_, color="b", ci=0.01)

        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="lower right", frameon=True)

        sns.despine(left=True, bottom=True)

        self.__show_or_save()

        self.__reset_figure_params()

    def add_figure_text(self, x, y, text):
        """
        给图像, 增加TEXT标注

        :param x: 标注图像上的X坐标值
        :param y: Y坐标值
        :param text: 标注内容
        :return:
        """
        # dict = {text: [x, y]}
        self.__text_dict.update({text: [x, y]})

    def __show_or_save(self, png_name=None):
        """
        展示或者保存图片
        :param png_name:
        :return:
        """

        if IS_LINUX:

            fig_path = self.__fig_path + '/%s.png' % png_name

            plt.savefig(fig_path, dpi=self.dpi)

        else:

            plt.show()

    @property
    def is_spline(self):
        return self.__is_spline

    @is_spline.setter
    def is_spline(self, is_spline):
        self.__is_spline = is_spline

    @property
    def x_label(self):
        return self.__x_label

    @x_label.setter
    def x_label(self, x_label):
        self.__x_label = x_label

    @property
    def y_label(self):
        return self.__y_label

    @y_label.setter
    def y_label(self, y_label):
        self.__y_label = y_label

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, title):
        self.__title = title

    @property
    def x_lim(self):
        return self.__x_lim

    @x_lim.setter
    def x_lim(self, x_lim):
        """
        :param x_lim:  X轴的【min, max】
        :return:
        """
        self.__x_lim = x_lim

    @property
    def y_lim(self):
        return self.__y_lim

    @y_lim.setter
    def y_lim(self, y_lim):
        self.__y_lim = y_lim

    @property
    def font(self):
        return self.__font

    @font.setter
    def font(self, font):
        self.__font = font

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, length):
        self.__length = length

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, size):
        self.__size = size

    @property
    def style(self):
        return self.__style

    @style.setter
    def style(self, style):
        self.__style = style

    @property
    def palette(self):
        return self.__palette

    @palette.setter
    def palette(self, palette):
        self.__palette = palette

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, width):
        self.__width = width

    @property
    def fig_path(self):
        return self.__fig_path

    @fig_path.setter
    def fig_path(self, fig_path):
        self.__fig_path = fig_path

    @property
    def dpi(self):
        return self.__dpi

    @dpi.setter
    def dpi(self, dpi):
        self.__dpi = dpi

    @property
    def spline_point_nums(self):
        return self.__spline_point_nums

    @spline_point_nums.setter
    def spline_point_nums(self, spline_point_nums):
        self.__spline_point_nums = spline_point_nums


    # def word_clound(self, inputText, inputJpgPath=None, pngName='wordcloud'):
    #     """
    #     词云图[基于分词词频]
    #
    #     inputJpgPath : 背景图片的全路径
    #
    #     """
    #
    #     pwd = path.dirname(__file__)
    #
    #     word_generator = jieba.cut(inputText) # 搜索模式的分词迭代器
    #
    #     word_list = ' '.join(word_generator) # 过滤其中为空的内容
    #
    #     # alice_coloring = plt.imread("%s/alice_color.jpg" % '/mnt/c/Users/HD/Desktop/nutsCloud/codes/python/DataMining/Chart')
    #
    #     alice_coloring = inputJpgPath if inputJpgPath else np.array(Image.open(path.join(pwd, ALICE_PNG_NAME))) # 是否指定背景图片
    #
    #     my_wordcloud = WordCloud(background_color="white",
    #                              max_words=100, mask=alice_coloring,
    #                              stopwords=STOPWORDS.add("said"),
    #                              max_font_size=50, random_state=42,
    #                              font_path=FIGURE_FILE_PHAT,
    #                              scale=2)
    #
    #     my_wordcloud.generate(word_list)
    #
    #     my_wordcloud.recolor(color_func=ImageColorGenerator(alice_coloring))
    #
    #     plt.axis("off")
    #
    #     if IS_LINUX: # linux 环境保存图片
    #
    #         my_wordcloud.to_file(path.join(pwd, "%s.png" % pngName))
    #
    #     else:
    #
    #         plt.imshow(my_wordcloud)
    #
    #         self.__show_or_save()


if __name__ == '__main__':

    df_iris = sns.load_dataset("iris")

    print(df_iris['species'].head())

