from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
import logging
from functools import wraps

logger = logging.getLogger(__name__)
auth_bp = Blueprint('auth', __name__)

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login_page():
    """Handle user login"""
    if request.method == 'GET':
        # Return login form (you can create a simple template)
        return '''
        <form method="post">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        '''
    
    # Handle login POST
    username = request.form.get('username')
    password = request.form.get('password')
    
    # TODO: Implement actual authentication
    # For now, accept any username/password
    if username and password:
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('index'))
    
    return "Invalid credentials", 401

@auth_bp.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    return redirect(url_for('auth.login_page'))

@auth_bp.route('/status')
def auth_status():
    """Check authentication status"""
    return jsonify({
        'logged_in': session.get('logged_in', False),
        'username': session.get('username')
    })