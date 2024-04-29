package smos.application.registerManagement;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Date;
import java.util.GregorianCalendar;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import smos.Environment;
import smos.bean.Classroom;
import smos.bean.RegisterLine;
import smos.bean.User;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.storage.ManagerClassroom;
import smos.storage.ManagerRegister;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;
import smos.utility.Utility;

public class ServletShowRegister extends HttpServlet {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4054623648928396283L;

	/**
	 * Definition of the doGet method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doGet(HttpServletRequest pRequest, 
			HttpServletResponse pResponse) {
		String gotoPage = "./registerManagement/showRegister.jsp";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		ManagerUser managerUser = ManagerUser.getInstance();
		User loggedUser = (User) session.getAttribute("loggedUser");
		ManagerClassroom managerClassroom = ManagerClassroom.getInstance();
		ManagerRegister  managerRegister= ManagerRegister.getInstance();
		Collection<RegisterLine> register = null;
		int year;
		int month;
		int day;
		
		String date=pRequest.getParameter("date");
		int idClass = Integer.valueOf(pRequest.getParameter("idClassroom"));
		
		String [] datevalues;
		datevalues = date.split("/");
		year = Integer.valueOf(datevalues[2]);
		month = Integer.valueOf(datevalues[1]);
		day = Integer.valueOf(datevalues[0]);
		
		try {
			register= managerRegister.getRegisterByClassIDAndDate(idClass, Utility.String2Date(date));
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			} 
			if ((!managerUser.isAdministrator(loggedUser)) && (!managerUser.isDirector(loggedUser))) {
				errorMessage =  "The logged in User does not have access to the " +
						"functionality'!";
				gotoPage = "./error.jsp";
			} 
			//set things to pass to the session, use session.setAttribute (String, attribute)
			
			
			
			Classroom classroom= managerClassroom.getClassroomByID(idClass);
			
			session.setAttribute("register", register);
			session.setAttribute("year", year);
			session.setAttribute("month", month);
			session.setAttribute("day", day);
			
			
			session.setAttribute("classroom", classroom);
			
			//take the academic year from the session
			pResponse.sendRedirect(gotoPage);
			return; 
			
		} catch (SQLException sqlException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + sqlException.getMessage();
			gotoPage = "./error.jsp";
			sqlException.printStackTrace();
		} catch (EntityNotFoundException entityNotFoundException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + entityNotFoundException.getMessage();
			gotoPage = "./error.jsp";
			entityNotFoundException.printStackTrace();
		} catch (ConnectionException connectionException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + connectionException.getMessage();
			gotoPage = "./error.jsp";
			connectionException.printStackTrace();
		}catch (IOException ioException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		}catch (InvalidValueException invalidValueException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + invalidValueException.getMessage();
			gotoPage = "./error.jsp";
			invalidValueException.printStackTrace();
		}
		
		session.setAttribute("errorMessage", errorMessage);
		try {
			pResponse.sendRedirect(gotoPage);
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		}
	}

	/**
	 * Definition of the doPost method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doPost(HttpServletRequest pRequest, 
			HttpServletResponse pResponse) {
		this.doGet(pRequest, pResponse);
	}
	
	
}
