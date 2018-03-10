import pandas as pd
import numpy as np
def AddBaseTimeFeature(df):

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    # df = df.drop(['date', 'time_interval'], axis=1)
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    # df['time_interval_year'] = df['time_interval_begin'].map(lambda x: x.strftime('%Y'))
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    return df
if __name__ == '__main__':
    sub_demo = pd.read_table(u'./semifinal_gy_cmp_testing_template_seg2.txt', header=None, sep=';')

    sub_demo.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
    sub_demo = sub_demo.sort_values(['link_ID', 'time_interval']).reset_index()
    del sub_demo['index']
    del sub_demo['travel_time']
    sub_demo = AddBaseTimeFeature(sub_demo)

    sub_demo_8 = sub_demo[sub_demo['time_interval_begin_hour']=='08']
    sub_demo_8 = sub_demo_8.reset_index()
    print sub_demo_8
    c_1 = pd.read_table(u'./2017-09-15_08_xgb.txt', header=None, sep='#')
    c_1.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
    print c_1[:122760]['travel_time']
    a = pd.concat([sub_demo_8[['link_ID','date','time_interval']],c_1[:122760]['travel_time']],axis=1)

    sub_demo_15 = sub_demo[sub_demo['time_interval_begin_hour'] == '15']
    sub_demo_15 = sub_demo_15.reset_index()
    print sub_demo_15
    c_2 = pd.read_table(u'./2017-09-15_15_xgb.txt', header=None, sep='#')
    c_2.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
    print c_2[:122760]['travel_time']
    b = pd.concat([sub_demo_15[['link_ID', 'date', 'time_interval']], c_2[:122760]['travel_time']], axis=1)


    sub_demo_18 = sub_demo[sub_demo['time_interval_begin_hour'] == '18']
    sub_demo_18 = sub_demo_18.reset_index()
    print sub_demo_18
    c_3 = pd.read_table(u'./2017-09-15_18_xgb.txt', header=None, sep='#')
    c_3.columns = ['link_ID', 'date', 'time_interval', 'travel_time']
    print c_3[:122760]['travel_time']
    c = pd.concat([sub_demo_18[['link_ID', 'date', 'time_interval']], c_3[:122760]['travel_time']], axis=1)


    # res = pd.concat([a,b,c])

    a.to_csv('./mapodoufu_2017-09-15_a_xgb.txt', sep='#', index=False,
                                                                   )
    print a[['link_ID', 'date', 'time_interval', 'travel_time']].shape
    print a[['link_ID', 'date', 'time_interval', 'travel_time']].isnull().sum()

    b.to_csv('./mapodoufu_2017-09-15_b_xgb.txt', sep='#', index=False,
            )
    print b[['link_ID', 'date', 'time_interval', 'travel_time']].shape
    print b[['link_ID', 'date', 'time_interval', 'travel_time']].isnull().sum()

    c.to_csv('./mapodoufu_2017-09-15_c_xgb.txt', sep='#', index=False,
             )
    print c[['link_ID', 'date', 'time_interval', 'travel_time']].shape
    print c[['link_ID', 'date', 'time_interval', 'travel_time']].isnull().sum()
