import { createContext, useContext, useState } from 'react';
import * as api from '../api/api.js';

const AuthContext = createContext(null);

function loadSession() {
  try {
    return JSON.parse(localStorage.getItem('auth_user'));
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(loadSession);

  const saveSession = (u) => {
    setUser(u);
    localStorage.setItem('auth_user', JSON.stringify(u));
  };

  const login = async (email, password) => {
    saveSession(await api.login(email, password));
  };

  const signup = async (name, email, password) => {
    saveSession(await api.signup(name, email, password));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('auth_user');
  };

  return (
    <AuthContext.Provider value={{ user, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
