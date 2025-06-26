import matplotlib.pyplot as plt
import numpy as np

# 假設有兩組資料：Plant1 與 Plant2
labels = ['Plant1', 'Plant2']
train_counts = [2611, 652]     # 藍色柱狀圖
test_counts = [653, 2612]      # 橘色柱狀圖

x = np.arange(len(labels))     # 資料集索引
width = 0.35                   # 柱狀圖寬度

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, train_counts, width, label='Train', color='royalblue')
rects2 = ax.bar(x + width/2, test_counts, width, label='Test', color='darkorange')

# 標題與軸標籤
ax.set_title('Train vs Test Sample Distribution')
ax.set_xlabel('Dataset')
ax.set_ylabel('Number of Samples')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(title='Data Type')

# 數值標籤
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.savefig( r'資料切分.png', bbox_inches='tight')
plt.show()
plt.close()
