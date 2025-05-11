import pandas as pd
import matplotlib.pyplot as plt

def one_plot(
            title:str,
            feature:str,
            df:pd.DataFrame,
            row:int,
            save_path=None,  # 저장 경로 매개변수 추가
    ):
        x = list(range(24))
        y = [df[f'{feature}_{n}'].iloc[row] for n in x]

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker='o', color='tab:blue')
        plt.title(f'{title} - Row {row}')
        plt.xlabel('Hour')
        plt.ylabel(f'{feature}')
        plt.grid(True)
        plt.xticks(x)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"이미지가 {save_path}에 저장되었습니다.")

        plt.show()


def comparison_plot(
        title:str,
        feature1:str,
        feature2:str,
        train_df1:pd.DataFrame,
        train_df2:pd.DataFrame,
        row:int,
        save_path=None):  # 저장 경로 매개변수 추가
    x = list(range(24))
    y1 = [train_df1[f'{feature1}_{n}'].iloc[row] for n in x]
    y2 = [train_df2[f'{feature2}_{n}'].iloc[row] for n in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # 왼쪽: 원본
    axes[0].plot(x, y1, marker='o', color='tab:blue')
    axes[0].set_title(f'Row {row}')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel(f'{feature1}')
    axes[0].grid(True)
    axes[0].set_xticks(x)

    # 오른쪽: 결측치 처리된 데이터
    axes[1].plot(x, y2, marker='o', color='tab:green')
    axes[1].set_title(f'{row}')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel(f'{feature2}')
    axes[1].grid(True)
    axes[1].set_xticks(x)

    plt.suptitle(f'{title}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"이미지가 {save_path}에 저장되었습니다.")

    plt.show()