# 🔐 Authentication System

## Overview
The Power Theft Detection System now includes admin authentication to protect access to the dashboard.

## Login Credentials

### Default Admin Account
- **Username:** `admin`
- **Password:** `admin123`

## Features

### ✅ Implemented Security Features
1. **Session-based Authentication** - Secure session management using Flask sessions
2. **Login Page** - Beautiful, modern login interface with demo credentials displayed
3. **Protected Routes** - All dashboard and API endpoints require authentication
4. **Logout Functionality** - Logout button in the navbar to end session
5. **Auto-redirect** - Unauthenticated users are automatically redirected to login

### 🔒 Protected Endpoints
- `/` - Main dashboard (requires login)
- `/api/available-years` - Available years API (requires login)
- `/api/year-statistics/<year>` - Year statistics API (requires login)
- `/api/year-consumption/<year>` - Consumption data API (requires login)
- `/api/year-detections/<year>` - Detection results API (requires login)

### 🌐 Public Endpoints
- `/login` - Login page (accessible to all)
- `/logout` - Logout endpoint (accessible to all)

## How to Use

### First Time Access
1. Navigate to `http://localhost:8105`
2. You will be automatically redirected to the login page
3. Enter the credentials:
   - Username: `admin`
   - Password: `admin123`
4. Click "🚀 Access Dashboard"
5. You will be redirected to the main dashboard

### Logging Out
1. Click the "🚪 Logout" button in the top-right corner of the navbar
2. You will be redirected back to the login page
3. Your session will be cleared

## Customization

### Adding More Users
Edit the `ADMIN_CREDENTIALS` dictionary in `app.py`:

```python
ADMIN_CREDENTIALS = {
    'admin': 'admin123',
    'user1': 'password1',
    'user2': 'password2'
}
```

### Changing the Secret Key
For production use, change the secret key in `app.py`:

```python
app.secret_key = 'your-secure-random-secret-key-here'
```

Generate a secure secret key using:
```python
import secrets
print(secrets.token_hex(32))
```

## Security Notes

⚠️ **Important for Production:**
1. **Change the default credentials** - Never use default credentials in production
2. **Use strong passwords** - Implement password complexity requirements
3. **Hash passwords** - Store hashed passwords instead of plain text (use `werkzeug.security`)
4. **Use HTTPS** - Always use HTTPS in production
5. **Implement rate limiting** - Prevent brute force attacks
6. **Add CSRF protection** - Use Flask-WTF for form protection
7. **Session timeout** - Implement automatic session expiration
8. **Audit logging** - Log all authentication attempts

## Future Enhancements

### Recommended Improvements
- [ ] Password hashing with bcrypt or argon2
- [ ] Database-backed user management
- [ ] Role-based access control (RBAC)
- [ ] Two-factor authentication (2FA)
- [ ] Password reset functionality
- [ ] Account lockout after failed attempts
- [ ] Session timeout and refresh
- [ ] Remember me functionality
- [ ] Activity logging and audit trail

## Technical Details

### Session Management
- Sessions are stored server-side
- Session cookie is HTTP-only (not accessible via JavaScript)
- Session expires when browser is closed (unless "remember me" is implemented)

### Authentication Flow
1. User visits protected route
2. `@login_required` decorator checks for session
3. If not authenticated, redirect to `/login`
4. User submits credentials
5. Server validates credentials
6. If valid, create session and redirect to dashboard
7. If invalid, show error and remain on login page

### Code Structure
- **app.py** - Main application with authentication logic
- **templates/login.html** - Login page template
- **templates/index_with_timeline.html** - Protected dashboard (with logout button)

## Support

If you encounter any issues with authentication:
1. Clear your browser cookies and cache
2. Restart the Flask server
3. Check the console for error messages
4. Verify credentials are correct

---

**Last Updated:** October 24, 2025  
**Version:** 1.0.0
