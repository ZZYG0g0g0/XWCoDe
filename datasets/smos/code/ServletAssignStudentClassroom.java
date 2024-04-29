package smos.application.userManagement;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import smos.Environment;
import smos.bean.Classroom;
import smos.bean.Role;
import smos.bean.User;
import smos.bean.UserListItem;
import smos.exception.DeleteManagerException;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.exception.MandatoryFieldException;
import smos.storage.ManagerClassroom;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;

public class ServletAssignStudentClassroom extends HttpServlet {

	/**
	 * 
	 */

	protected void doGet(HttpServletRequest pRequest,
			HttpServletResponse pResponse) {
		String gotoPage = "./showUserList";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		ManagerUser managerUser = ManagerUser.getInstance();
		ManagerClassroom managerClassroom = ManagerClassroom.getInstance();
		User loggedUser = (User) session.getAttribute("loggedUser");

		// Verify that the user is logged in
		try {
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			}
			if ((!managerUser.isAdministrator(loggedUser))) {
				errorMessage = "The logged-in User does not have access to "
					+ "functionality'!";
				gotoPage = "./error.jsp";
			}

			//Selected student
			User user = (User) session.getAttribute("user");

			//selected academic year
			int academicYear = Integer.valueOf(pRequest
					.getParameter("accademicYear"));
			// id selected class which is zero if we want to delete
			//The association
			int idClassroom = Integer.valueOf(pRequest.getParameter("classId"));
			Classroom p = null;
			p = managerClassroom.getClassroomByUserAcademicYear(user,academicYear);
			if (p != null)
				managerUser.removeStudentClassroom(user, p.getIdClassroom());
			if(idClassroom>0) {
				
					managerUser.assignStudentClassroom(user, idClassroom);
			}

		} catch (NumberFormatException numberFormatException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
			+ numberFormatException.getMessage();
			gotoPage = "./error.jsp";
			numberFormatException.printStackTrace();
		} catch (EntityNotFoundException entityNotFoundException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
			+ entityNotFoundException.getMessage();
			gotoPage = "./error.jsp";
			entityNotFoundException.printStackTrace();
		} catch (SQLException SQLException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
			+ SQLException.getMessage();
			gotoPage = "./error.jsp";
			SQLException.printStackTrace();
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
		} catch (InvalidValueException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (MandatoryFieldException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		session.setAttribute("errorMessage", errorMessage);
		try {
			pResponse.sendRedirect(gotoPage);
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE
			+ ioException.getMessage();
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