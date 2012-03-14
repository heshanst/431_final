#include "Matrix.hpp"
#include <numeric>

class MatrixUtil {
public:
    typedef double (*F)(double);

    static bool is_almost_symmetric(const Matrix& A, double ap=1e-6, double rp=1e-4) {
        if (A.rows() != A.cols()) {
            return false;
        }

        for (unsigned int r = 0; r < A.rows() - 1; ++r) {
            for (unsigned int c = 0; c < r; c++) {
                double delta = abs(A[r][c] - A[c][r]);
                if (delta > ap && delta > std::max(abs(A[r][c]), abs(A[c][r])) * rp) {
                    return false;
                }
            }
        }

        return true;
    }

    static bool is_almost_zero(const Matrix& A, double ap=1e-6, double rp=1e-4) {
        for (unsigned int r = 0; r < A.rows(); ++r) { 
            for (unsigned int c = 0; c < A.cols(); ++c) { 
                double delta = abs(A[r][c]-A[c][r]);
                if (delta>ap && delta>std::max(abs(A[r][c]),abs(A[c][r]))*rp) {
                    return false;
                }
            }
            return true;
        }
    }

    static double norm(double A, int p=1) {
        return abs(A);
    }

    static double norm(std::vector<double>& A, int p=1) {
        double x = 0.0;
        for (unsigned int i = 0; i < A.size(); ++i) {
          x += pow(A[i],p);
        }
        return pow(x, 1.0/p);
    }

    static double norm(const Matrix& A, int p=1) {
        if (A.rows() == 1 || A.cols() == 1) {
            double x = 0;
            for (unsigned int r = 0; r < A.rows(); ++r) {
                for (unsigned int c = 0; c < A.cols(); ++c) {
                    x += pow(norm(A[r][c]), p);
                }
            }
            return pow(x,1.0/p);
        } else if (p==1) {
            double mx = 0;
            for (unsigned int r = 0; r < A.rows(); ++r) {
                double x = 0;
                for (unsigned int c = 0; c < A.cols(); ++c) {
                    x += norm(A[r][c]);
                }
                if (x > mx) {
                    mx = x;
                }
            }
            return mx;
        }
        else {
            throw std::runtime_error("error");
        }
    }

    static double condition_number(double (*f)(double), double x=0.0, double h=1e-6) {
        //return D(f,h)(x)*x/f(x);
        return f(x) * x/f(x);
    }

    static double condition_number(const Matrix& f, double x=0.0, double h=1e-6) {
        //if is the Matrix J
        return norm(f)*norm(1/f);
    }

    static Matrix exp(const Matrix& x,double ap=1e-6, double rp=1e-4, int ns=40) {
        Matrix t  = Matrix::identity(x.cols());
        Matrix s(t);
        for (int k = 1; k < ns; ++k) {
            t = t*x/k;   // next term
            s = s + t;   // add next term
            if (norm(t)<std::max(ap,norm(s)*rp)) {
                return s;
            }
        }
        throw std::runtime_error("error");
    }

    static Matrix Cholesky(const Matrix& A) {
        if (!is_almost_symmetric(A)) {
            throw std::runtime_error("error");
        }
        Matrix L(A);
        for (unsigned int k = 0; k < L.cols(); ++k) {
            if (L[k][k]<=0) {
                throw std::runtime_error("error");
            }
            double p = L[k][k] = sqrt(L[k][k]);
            for (unsigned int i = k+1; i < L.rows(); ++i) {
                L[i][k] /= p;
            }
            for (unsigned int j = k + 1; j < L.rows(); ++j) {
                double p=double(L[j][k]);
                for (unsigned int i = k+1; i<L.rows(); ++i) {
                    L[i][j] -= p*L[i][k];
                }
            }
        }
        for (unsigned int i = 0; i < L.rows(); ++i) {
            for (unsigned int j = i+1; j < L.cols(); ++j) {
                L[i][j]=0;
            }
        }
        return L;
    }

    static bool is_positive_definite(const Matrix& A) {
        if (!is_almost_symmetric(A)) {
            return false;
        }
        try {
            Cholesky(A);
            return true;
        } catch (...) {
            return false;
        }
    }

    //Computes c_j for best linear fit of y[i] \pm dy[i] = fitting_f(x[i])
    //where fitting_f(x[i]) is \sum_j c_j f[j](x[i])
    //parameters:
    //- a list of fitting functions
    //- a list with points (x,y,dy)
    //returns:
    //- column vector with fitting coefficients
    //- the chi2 for the fit
    static Matrix fit_least_squares(const std::vector<std::vector<double> >& points, 
        const std::vector<double (*)(double)>& f,
        double& chi2) {

            Matrix A = Matrix(points.size(), f.size());
            Matrix b = Matrix(points.size(), points.size());

            for (unsigned int i = 0; i < A.rows(); ++i) {
                double weight = points[i].size() > 2 ? 1.0/points[i][2] : 1.0;
                b[i][0] = weight*double(points[i][1]);
                for (unsigned int j = 0; j < A.cols(); ++j) {
                    A[i][j] = weight*f[j](points[i][0]);
                }
            }
            Matrix c = (1.0/(A.t()*A))*(A.t()*b);
            Matrix chi = A*c-b;
            chi2 = pow(norm(chi,2), 2);

            return c;
    }

    static double solve_fixed_point(F f, double x, double ap=1e-6, double rp=1e-4, int ns=100) {

        for (int k = 0; k < ns; ++k) {
            if (abs(x) >= 1) {
                throw std::runtime_error("error");
            }
            double x_old = x;
            x = f(x)+x;
            if (k>2 && norm(x_old-x)<std::max(ap,norm(x)*rp)) {
                return x;
            }
        }
        throw std::runtime_error("no convergence");
    }

    static double solve_bisection(double (*f)(double), double a, double b, double ap=1e-6, double rp=1e-4, int ns=100) {
        double fa = f(a);
        double fb = f(b);
        if (fa == 0) {
            return a;
        }
        if (fb == 0) {
            return b;
        }
        if (fa*fb > 0) {
            throw std::runtime_error("f(a) and f(b) must have opposite sign");
        }
        for (int k = 0; k < ns; ++k) {
            double x = (a+b)/2;
            double fx = f(x);
            if (fx==0 || norm(b-a)<std::max(ap,norm(x)*rp)) {
                return x;
            } else if (fx * fa < 0) {
                b = x;
                fb = fx;
            } else {
                a = x;
                fa = fx;
            }
        }
        throw std::runtime_error("no convergence");
    }


    static double solve_newton(double (*f)(double), double x, double ap=1e-6, double rp=1e-4, int ns=20) {
        for (int k = 0; k < ns; ++k) {
            double fx=f(x),dfx = D(f)(x);
            if (norm(dfx) < ap) {
                throw std::runtime_error("unstable solution");
            }
            double x_old = x;
            x = x-fx/dfx;
            if (k>2 && norm(x-x_old)<std::max(ap,norm(x)*rp)) {
                return x;
            }
        }

        throw std::runtime_error("no convergence");
    }

    static double solve_secant(F f, double x, double ap=1e-6, double rp=1e-4, int ns=20) {
        double fx = f(x), dfx = D(f)(x);
        for (int k = 0; k < ns; ++k) {
            if (norm(dfx) < ap) {
                throw std::runtime_error("unstable solution");
            }
            double x_old = x;
            double fx_old = fx;
            x = x - fx/dfx;
            if (k>2 && norm(x-x_old)<std::max(ap,norm(x)*rp)) {
                return x;
            }
            fx = f(x);
            dfx = (fx-fx_old)/(x-x_old);
        }
        throw std::runtime_error("no convergence");
    }

    static double solve_newton_stabilized(F f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=20) {
        double fa = f(a), fb = f(b);
        if (fa == 0) {
            return a;
        }
        if (fb == 0){
            return b;
        }
        if (fa*fb > 0) {
            throw std::runtime_error("f(a) and f(b) must have opposite sign");
        }
        double x = (a+b)/2;
        double fx = f(x), dfx = D(f)(x);
        for (int k = 0; k < ns; ++k) {
            double x_old = x, fx_old = fx;
            if (norm(dfx)>ap) {
                x = x - fx/dfx;
            }
            if (x==x_old || x<a || x>b) {
                x = (a+b)/2;
            }
            fx = f(x);
            if (fx==0 || norm(x-x_old)<std::max(ap,norm(x)*rp)) {
                return x;
            }
            dfx = (fx-fx_old)/(x-x_old);
            if (fx * fa < 0) {
                b = x,fb = fx;
            } else {
                a = x, fa =  fx;
            }
        }
        throw std::runtime_error("no convergence");
    }

    static double optimize_bisection(F f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=100) {
        double Dfa = D(f)(a), Dfb = D(f)(b);
        if (Dfa == 0 ) {
            return a;
        }
        if (Dfb == 0) {
            return b;
        }
        if (Dfa*Dfb > 0) {
            throw std::runtime_error("D(f)(a) and D(f)(b) must have opposite sign");
        }
        for (int k = 0; k < ns; ++k) {
            double x = (a+b)/2;
            double Dfx = D(f)(x);
            if (Dfx==0 || norm(b-a)<std::max(ap,norm(x)*rp)) {
                return x;
            } else if (Dfx * Dfa < 0) {
                b = x,Dfb = Dfx;
            } else {
                a = x,Dfa = Dfx;
            }
        }

        throw std::runtime_error("no convergence");
    }
    static double optimize_newton(F f, double x, double ap=1e-6, double rp=1e-4, int ns=20) {
        for (int k = 0; k < ns; ++k) {
            double Dfx = D(f)(x), DDfx = DD(f)(x);
            if (Dfx==0 ) {
                return x;
            }
            if (norm(DDfx) < ap) {
                throw std::runtime_error("unstable solution");
            }
            double x_old = x, x = x-Dfx/DDfx;
            if (norm(x-x_old)<std::max(ap,norm(x)*rp)){
                return x;
            }
        }
        throw std::runtime_error("no convergence");
    }

    static double optimize_secant(F f, double x, double ap=1e-6, double rp=1e-4, int ns=100) {
        double fx = f(x), Dfx = D(f)(x), DDfx = DD(f)(x);
        for (int k = 0; k < ns; ++k) {
            if (Dfx==0 ) {
                return x;
            }
            if (norm(DDfx) < ap) {
                throw std::runtime_error("unstable solution");
            }
            double x_old = x, Dfx_old = Dfx, x = x-Dfx/DDfx;
            if (norm(x-x_old)<std::max(ap,norm(x)*rp)) {
                return x;
            }
            fx = f(x);
            Dfx = D(f)(x);
            DDfx = (Dfx - Dfx_old)/(x-x_old);
        }

        throw std::runtime_error("no convergence");
    }

    static double optimize_newton_stabilized(F f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=20) {
        double Dfa = D(f)(a), Dfb = D(f)(b);
        if (Dfa == 0) {
            return a;
        }
        if (Dfb == 0) {
            return b;
        }
        if (Dfa*Dfb > 0) {
            throw std::runtime_error("D(f)(a) and D(f)(b) must have opposite sign");
        }
        double x = (a+b)/2;
        double fx = f(x), Dfx = D(f)(x), DDfx = DD(f)(x);
        for (int k = 0; k < ns; ++k) {
            if (Dfx==0){ 
                return x;
            }
            double x_old = x, fx_old = fx, Dfx_old = Dfx;
            if (norm(DDfx)>ap) {
                x = x - Dfx/DDfx;
            }
            if (x==x_old || x<a || x>b) {
                x = (a+b)/2;
            }
            if (norm(x-x_old)<std::max(ap,norm(x)*rp)) {
                return x;
            }
            fx = f(x);
            Dfx = (fx-fx_old)/(x-x_old);
            DDfx = (Dfx-Dfx_old)/(x-x_old);
            if (Dfx * Dfa < 0) {
                b = x, Dfb = Dfx;
            } else {
                a = x,Dfa = Dfx;
            }
        }

        throw std::runtime_error("no convergence");
    }

    static double optimize_golden_search(F f, double a, double b, double ap=1e-6, double rp=1e-4, int ns=100) {
        double tau = (sqrt(5.0)-1.0)/2.0;
        double x1 = a+(1.0-tau)*(b-a), x2= a+tau*(b-a);
        double fa = f(a), f1 = f(x1), f2 = f(x2), fb = f(b);
        for (int k =0; k < ns; ++k ) {
            if (f1 > f2) {
                a = x1, fa = f1, x1 = x2, f1 = f2;
                x2 = a+tau*(b-a);
                f2 = f(x2);
            } else {
                b = x2, fb = f2, x2 = x1, f2 = f1;
                x1 = a+(1.0-tau)*(b-a);
                f1 = f(x1);
            }
            if (k>2 && norm(b-a)<std::max(ap,norm(b)*rp)) {
                return b;
            }
        }

        throw std::runtime_error("no convergence");
    }

    static F D(F f) {
        return f;
    }

    static F DD(F f) {
        return f;
    }
};
