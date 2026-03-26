// DeDupe ML Framework - JavaScript Functions
// Copy this entire file to: static/js/scripts.js

console.log('Scripts.js loaded successfully');

// Auth Functions
function toggleAuthMode(mode) {
    console.log('toggleAuthMode called with:', mode);
    const btnSignin = document.getElementById('tab-signin');
    const btnSignup = document.getElementById('tab-signup');
    const formSignin = document.getElementById('form-signin');
    const formSignup = document.getElementById('form-signup');

    if (!btnSignin || !btnSignup || !formSignin || !formSignup) {
        console.error('One or more elements not found');
        return;
    }

    if (mode === 'signin') {
        btnSignin.classList.add('bg-accent-500', 'text-white', 'shadow-md');
        btnSignin.classList.remove('text-gray-400');
        btnSignup.classList.remove('bg-accent-500', 'text-white', 'shadow-md');
        btnSignup.classList.add('text-gray-400');
        
        formSignin.classList.remove('hidden');
        formSignup.classList.add('hidden');
    } else {
        btnSignup.classList.add('bg-accent-500', 'text-white', 'shadow-md');
        btnSignup.classList.remove('text-gray-400');
        btnSignin.classList.remove('bg-accent-500', 'text-white', 'shadow-md');
        btnSignin.classList.add('text-gray-400');

        formSignup.classList.remove('hidden');
        formSignin.classList.add('hidden');
    }
}

// Handle Login
async function handleLogin(e) {
    e.preventDefault();
    console.log('Login function called');
    
    const emailInput = document.querySelector('#form-signin input[type="email"]');
    const passwordInput = document.querySelector('#form-signin input[type="password"]');
    
    if (!emailInput || !passwordInput) {
        console.error('Email or password input not found');
        return;
    }
    
    const email = emailInput.value;
    const password = passwordInput.value;
    
    console.log('Attempting login with:', email);
    
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        console.log('Login response:', data);
        
        if (data.success) {
            console.log('Login successful, redirecting...');
            window.location.href = '/dashboard';
        } else {
            alert(data.message || 'Login failed. Please check your credentials.');
        }
    } catch (error) {
        console.error('Login error:', error);
        alert('An error occurred during login. Please try again.');
    }
}

// Handle Signup  
async function handleSignup(e) {
    e.preventDefault();
    console.log('Signup function called');
    
    const nameInput = document.querySelector('#form-signup input[type="text"]');
    const emailInput = document.querySelector('#form-signup input[type="email"]');
    const passwords = document.querySelectorAll('#form-signup input[type="password"]');
    
    if (!nameInput || !emailInput || passwords.length < 2) {
        console.error('Form inputs not found');
        alert('Please fill in all fields');
        return;
    }
    
    const name = nameInput.value;
    const email = emailInput.value;
    const password = passwords[0].value;
    const confirmPassword = passwords[1].value;
    
    console.log('Attempting signup with:', name, email);
    
    // Validate passwords match
    if (password !== confirmPassword) {
        alert('Passwords do not match!');
        return;
    }
    
    // Validate all fields are filled
    if (!name || !email || !password) {
        alert('Please fill in all fields');
        return;
    }
    
    try {
        const response = await fetch('/api/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, email, password })
        });
        
        const data = await response.json();
        console.log('Signup response:', data);
        
        if (data.success) {
            console.log('Signup successful, redirecting...');
            alert('Account created successfully! Redirecting to dashboard...');
            window.location.href = '/dashboard';
        } else {
            alert(data.message || 'Signup failed. Please try again.');
        }
    } catch (error) {
        console.error('Signup error:', error);
        alert('An error occurred during signup. Please try again.');
    }
}

// Handle Logout
function handleLogout() {
    console.log('Logging out...');
    window.location.href = '/logout';
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing...');
    
    // Check if we're on signin page
    const tabSignin = document.getElementById('tab-signin');
    const tabSignup = document.getElementById('tab-signup');
    
    if (tabSignin && tabSignup) {
        console.log('Setting up signin/signup page');
        
        // Initialize to signin mode
        toggleAuthMode('signin');
        
        // Add click handlers to tabs
        tabSignin.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Signin tab clicked');
            toggleAuthMode('signin');
        });
        
        tabSignup.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Signup tab clicked');
            toggleAuthMode('signup');
        });
    }
    
    // Add event listeners to forms
    const signinForm = document.getElementById('form-signin');
    const signupForm = document.getElementById('form-signup');
    
    if (signinForm) {
        console.log('Attaching signin form listener');
        signinForm.addEventListener('submit', handleLogin);
    }
    
    if (signupForm) {
        console.log('Attaching signup form listener');
        signupForm.addEventListener('submit', handleSignup);
    }
    
    // Add Google/GitHub button handlers (placeholder)
    const socialButtons = document.querySelectorAll('.grid.grid-cols-2 button');
    
    socialButtons.forEach(function(button) {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const text = button.textContent.trim();
            console.log('Social button clicked:', text);
            alert(text + ' sign-in coming soon!');
        });
    });
    
    console.log('Initialization complete');
});

// Mobile Nav Toggle
function toggleMobileNav() {
    const drawer = document.getElementById('mobile-nav');
    if (!drawer) return;
    drawer.classList.toggle('open');
}

// Close mobile nav when clicking outside
document.addEventListener('click', function(e) {
    const drawer = document.getElementById('mobile-nav');
    const hamburger = document.querySelector('.topbar-hamburger');
    if (drawer && drawer.classList.contains('open')) {
        if (!drawer.contains(e.target) && e.target !== hamburger && !hamburger?.contains(e.target)) {
            drawer.classList.remove('open');
        }
    }
});

// Make functions globally available
window.toggleAuthMode = toggleAuthMode;
window.handleLogin = handleLogin;
window.handleSignup = handleSignup;
window.handleLogout = handleLogout;
window.toggleMobileNav = toggleMobileNav;