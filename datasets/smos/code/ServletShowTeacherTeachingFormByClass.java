package smos.application.userManagement;

import smos.Environment;
import smos.bean.Classroom;
import smos.bean.Teaching;
import smos.bean.User;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.storage.ManagerClassroom;
import smos.storage.ManagerTeaching;
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
public class ServletShowTeacherTeachingFormByClass extends HttpServlet {

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
		String gotoPage = "./persistentDataManagement/userManagement/teacherTeachings.jsp";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		Collection<Teaching> teachingList = null;
		ManagerUser managerUser = ManagerUser.getInstance();
		User loggedUser = (User) session.getAttribute("loggedUser");
		//User teacher = (User) session.getAttribute("user");
		try {
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			}
			if (!managerUser.isAdministrator(loggedUser)) {
				errorMessage = "The logged in User does not have access to the "
						+ "functionality'!";
				gotoPage = "./error.jsp";
			}
			// we retrieve the id of the selected class
			int selectedClassId = Integer.valueOf(pRequest.getParameter("classId"));
			ManagerTeaching managerTeaching = ManagerTeaching.getInstance();
			//We calculate the list of courses associated with the selected class
			teachingList = managerTeaching.getTeachingsByClassroomId(selectedClassId);
			session.setAttribute("teachingList", teachingList);
			Classroom selectedClass = ManagerClassroom.getInstance().getClassroomByID(selectedClassId);
			session.setAttribute("selectedClass", selectedClass);
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
