from flask import Flask, request, jsonify, render_template
import os
import subprocess

app = Flask(__name__)

# 設定上傳資料夾
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    # 對應到 templates/index.html
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    # 檢查是否有收到 'files'
    if 'files' not in request.files:
        return jsonify({'error': '沒有上傳任何檔案'}), 400

    # 取得多個檔案（使用 getlist 以一次抓取所有 'files' 欄位）
    files = request.files.getlist('files')
    if not files or (len(files) == 1 and files[0].filename == ''):
        return jsonify({'error': '未選擇檔案或檔案名稱為空'}), 400

    saved_files = []
    for file in files:
        # 儲存檔案到指定資料夾
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        saved_files.append(file_path)

    # 呼叫 .bat 檔案 (process_media.bat) 進行檔名改名處理
    # 使用 cmd /c 來執行批次檔，並帶上 shell=True 保證在 Windows 下正確執行
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bat_path = os.path.join(current_dir, 'process_media.bat')

    try:
        result = subprocess.run(['cmd', '/c', bat_path],
                                capture_output=True, text=True, shell=True)
        stdout_text = result.stdout
        stderr_text = result.stderr

        # 如果 .bat 返回非 0，但檔案已被成功改名，你可以忽略這個錯誤。
        if result.returncode != 0:
            print("Warning: .bat returned non-zero exit code:", result.returncode)
            print("stdout:", stdout_text)
            print("stderr:", stderr_text)
            # 或者將返回碼設為 0 (如果你確定執行結果正確)
            # result.returncode = 0

    except Exception as e:
        return jsonify({'error': f'呼叫 .bat 發生錯誤: {str(e)}'}), 500

    return jsonify({
        'message': '上傳成功，已成功重新命名',
        'saved_files': saved_files,
        'stdout': stdout_text,
        'stderr': stderr_text
    }), 200

if __name__ == '__main__':
    # 讓 Flask 在 0.0.0.0:5000 啟動，這樣外部裝置也可連線
    app.run(host='0.0.0.0', port=5000, debug=True)
