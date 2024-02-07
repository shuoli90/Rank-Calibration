import utils.cd_diagram as cd

def plot_cd_diagram(df, title, save_dir='../tmp', col1='indicator', col2='metric', col3='score'):
    cd.draw_cd_diagram(df_perf=df, title=title, labels=True, save_dir=save_dir, col1=col1, col2=col2, col3=col3)