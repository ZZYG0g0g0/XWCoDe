package smos.application.userManagement;

import smos.Environment;
import smos.bean.Classroom;
import smos.bean.User;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.storage.ManagerClassroom;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Collection;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

/**
 * Servlet used to view all academic years present in the
 * db.
 * 
 * @author Giulio D'Amora
 * @version 1.0
 * 
 *          2009 ?Copyright by SMOS
 */
public class ServletLoadClassByAccademicYear extends HttpServlet {

	private static final long serialVersionUID = -3988115259356084996L;

	/**
	 * Definition of the doGet method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doGet(HttpServletRequest pRequest,
			HttpServletResponse pResponse) {
		String gotoPage = "./persistentDataManagement";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		Collection<Classroom> classroomList = null;
		ManagerUser managerUser = ManagerUser.getInstance();
		User loggedUser = (User) session.getAttribute("loggedUser");
		
		try {
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			}
			if ((!managerUser.isAdministrator(loggedUser)) && (!managerUser.isDirector(loggedUser))) {
				errorMessage = "The logged in User does not have access to the "
						+ "functionality'!";
				gotoPage = "./error.jsp";
			}
			//Date oggi = new Date();
			// we recover the selected academic year
			int selectedAccademicYear = Integer.valueOf(pRequest.getParameter("accademicYear"));
			
			
			ManagerClassroom managerClassroom = ManagerClassroom.getInstance();
			//We calculate the list of classes of the selected academic year
			classroomList = managerClassroom.getClassroomsByAcademicYear(selectedAccademicYear);
			session.setAttribute("classroomList", classroomList);
			session.setAttribute("selectedYear", selectedAccademicYear);
			//session.removeAttribute("selectedClass");
			gotoPage +=(String) session.getAttribute("goTo");
			
			pResponse.sendRedirect(gotoPage);
			return;

		} catch (SQLException sqlException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ sqlException.getMessage();
			gotoPage = "./error.jsp";
			sqlException.printStackTrace();
		} catch (EntityNotFoundException entityNotFoundException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ entityNotFoundException.getMessage();
			gotoPage = "./error.jsp";
			entityNotFoundException.printStackTrace();
		} catch (ConnectionException connectionException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ connectionException.getMessage();
			gotoPage = "./error.jsp";
			connectionException.printStackTrace();
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
		} catch (InvalidValueException invalidValueException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ invalidValueException.getMessage();
			gotoPage = "./error.jsp";
			invalidValueException.printStackTrace();
		}

		pRequest.getSession().setAttribute("errorMessage", errorMessage);
		try {
			pResponse.sendRedirect(gotoPage);
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
					+ ioException.getMessage();
			gotoPage = "./error6.jsp";
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
