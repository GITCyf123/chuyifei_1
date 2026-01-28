from flask import Flask, render_template, request, jsonify, send_from_directory
import datetime
import random
import os

app = Flask(__name__)

# 添加对jpg文件夹的静态文件访问支持
@app.route('/jpg/<path:filename>')
def serve_jpg(filename):
    return send_from_directory(os.path.join(app.root_path, 'jpg'), filename)

# 全局变量用于存储模拟状态
current_values = {
    'power_voltage': 0.0,
    'power_current': 0.0,
    'magnet_current': 0.0,
    'target_positions': {'内靶': 0, '剥离靶': 0},
    'ion_source_status': {'内部源1': False, '内部源2': True, '外部源1': False, '辅助源1': True}
}

@app.route('/')
def index():
    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    
    # 模拟数据 - 在实际应用中，这些数据会从设备或数据库获取
    data = {
        'main_title': '9.5MeV控制系统',
        'current_time': current_time,
        'instruments': {
            '内靶': '0.00',
            '剥离靶': '0.00',
            '引出法拉第筒': '0.00',
            '光栅1': '0.00'
        },
        'power_supply': {
            'voltage_set': '{0:.2f}'.format(current_values['power_voltage']),
            'current_set': '{0:.2f}'.format(current_values['power_current']),
            'voltage_feedback': '{0:.2f}'.format(current_values['power_voltage']),
            'current_feedback': '{0:.2f}'.format(current_values['power_current'])
        },
        'magnet_power': {
            'current_set': '{0:.2f}'.format(current_values['magnet_current']),
            'current_feedback': '{0:.2f}'.format(current_values['magnet_current']),
            'voltage_feedback': '{0:.2f}'.format(current_values['magnet_current'] * 10)
        },
        'targets': {
            '内靶': {'position': current_values['target_positions']['内靶']},
            '剥离靶': {'position': current_values['target_positions']['剥离靶']}
        },
        'water_cooling': [
            {'label': '水冷1路', 'status': 'normal'},
            {'label': '水冷2路', 'status': 'normal'},
            {'label': '水冷3路', 'status': 'normal'},
            {'label': '水冷4路', 'status': 'normal'},
            {'label': '水冷5路', 'status': 'normal'}
        ],
        'vacuum': {
            'low_pressure': {'value': '0.00', 'status': 'warning'},
            'high_pressure': {'value': '0.00', 'status': 'warning'},
            'molecular_pump1': {'frequency': '0.0 Hz', 'status': 'normal'},
            'molecular_pump2': {'frequency': '0.1 Hz', 'status': 'normal'}
        },
        'ion_sources': [
            {'label': '内部源1', 'current': '0.00', 'status': 'on' if current_values['ion_source_status']['内部源1'] else 'off'},
            {'label': '内部源2', 'current': '4.0', 'status': 'on' if current_values['ion_source_status']['内部源2'] else 'off'},
            {'label': '外部源1', 'current': '0.00', 'status': 'on' if current_values['ion_source_status']['外部源1'] else 'off'},
            {'label': '辅助源1', 'current': '5.0', 'status': 'on' if current_values['ion_source_status']['辅助源1'] else 'off'}
        ]
    }
    
    return render_template('index.html', data=data)

# API路由 - 处理控制面板的交互
@app.route('/api/update_value', methods=['POST'])
def update_value():
    """处理值的更新请求"""
    try:
        data = request.get_json()
        value_type = data.get('type')
        operation = data.get('operation')
        step = float(data.get('step', 0.1))
        
        if value_type == 'power_voltage':
            if operation == 'increase':
                current_values['power_voltage'] += step
            elif operation == 'decrease':
                current_values['power_voltage'] = max(0, current_values['power_voltage'] - step)
        elif value_type == 'power_current':
            if operation == 'increase':
                current_values['power_current'] += step
            elif operation == 'decrease':
                current_values['power_current'] = max(0, current_values['power_current'] - step)
        elif value_type == 'magnet_current':
            if operation == 'increase':
                current_values['magnet_current'] += step
            elif operation == 'decrease':
                current_values['magnet_current'] = max(0, current_values['magnet_current'] - step)
        
        return jsonify({
            'success': True,
            'value': round(current_values.get(value_type, 0), 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/target_position', methods=['POST'])
def target_position():
    """处理靶位移动请求"""
    try:
        data = request.get_json()
        target = data.get('target')
        direction = data.get('direction')
        
        if target in current_values['target_positions']:
            if direction == 'forward':
                current_values['target_positions'][target] += 1
            elif direction == 'backward':
                current_values['target_positions'][target] = max(0, current_values['target_positions'][target] - 1)
        
        return jsonify({
            'success': True,
            'position': current_values['target_positions'].get(target, 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/toggle_ion_source', methods=['POST'])
def toggle_ion_source():
    """切换离子源状态"""
    try:
        data = request.get_json()
        source = data.get('source')
        
        if source in current_values['ion_source_status']:
            current_values['ion_source_status'][source] = not current_values['ion_source_status'][source]
        
        return jsonify({
            'success': True,
            'status': 'on' if current_values['ion_source_status'].get(source, False) else 'off'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/quick_action', methods=['POST'])
def quick_action():
    """处理一键操作请求"""
    try:
        data = request.get_json()
        action = data.get('action')
        
        # 模拟一键操作的响应
        if action == '准备状态':
            # 模拟准备状态操作
            result = {'success': True, 'message': '系统进入准备状态'}
        elif action == '真空启动':
            # 模拟真空启动
            result = {'success': True, 'message': '真空系统已启动'}
        elif action == '真空关闭':
            # 模拟真空关闭
            result = {'success': True, 'message': '真空系统已关闭'}
        elif action == '起辉阶段':
            # 模拟起辉
            result = {'success': True, 'message': '离子源开始起辉'}
        elif action == '自动调节':
            # 模拟自动调节
            result = {'success': True, 'message': '系统开始自动调节'}
        elif action == '停止出束':
            # 模拟停止出束
            result = {'success': True, 'message': '已停止出束'}
        else:
            result = {'success': False, 'message': '未知操作'}
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_instruments', methods=['GET'])
def update_instruments():
    """获取最新的仪表数据（模拟实时更新）"""
    try:
        # 生成随机模拟数据
        instruments = {
            '内靶': round(random.uniform(0, 10), 2),
            '剥离靶': round(random.uniform(0, 10), 2),
            '引出法拉第筒': round(random.uniform(0, 10), 2),
            '光栅1': round(random.uniform(0, 10), 2)
        }
        
        return jsonify({
            'success': True,
            'instruments': instruments
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import socket
    
    # 检查端口5000是否被占用
    print("检查端口5000是否被占用...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 5000))
            print("端口5000可用，应用程序可以启动")
    except socket.error as e:
        print(f"警告: 端口5000可能已被占用，请先关闭占用该端口的程序")
        print(f"错误信息: {e}")
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)