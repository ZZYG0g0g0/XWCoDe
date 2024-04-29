package smos.application.userManagement;

import smos.Environment;
import smos.bean.Role;
import smos.bean.User;
import smos.bean.UserListItem;
import smos.exception.DeleteManagerException;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

/**
 * Servlet used to assign one or more roles to a user.
 * 
 * @author Napolitano Vincenzo.
 * 
 */
public class ServletAssignRole extends HttpServlet {

	private static final long serialVersionUID = 537330195407987283L;
	
	/**
	 * Definition of the doGet method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	protected void doGet(HttpServletRequest pRequest,
			HttpServletResponse pResponse) {
		String gotoPage = "showUserList";
		String errorMessage = "";
		HttpSession session = pRequest.getSession();
		
		Collection<UserListItem> administrators = new Vector<UserListItem>();
		Iterator<UserListItem> itAdmin = null;
		
		ManagerUser managerUser = ManagerUser.getInstance();

		User loggedUser = (User) session.getAttribute("loggedUser");

		//Verify that the user is logged in
		try {
			if (loggedUser == null) {
				pResponse.sendRedirect("./index.htm");
				return;
			}
			if ((!managerUser.isAdministrator(loggedUser))) {
				errorMessage =  "The logged-in User does not have access to " +
						"functionality'!";
				gotoPage = "./error.jsp";
			}
			
			User user = (User) session.getAttribute("user");
			
			administrators = managerUser.getAdministrators();
			itAdmin = administrators.iterator();
			itAdmin.next();
			
			String[] selectedRoles = pRequest.getParameterValues("selectedRoles");
			String[] unselectedRoles = pRequest.getParameterValues("unselectedRoles");
			
			if (selectedRoles != null) {
				int selectedlength = selectedRoles.length;
				for (int i = 0; i < selectedlength; i++) {
					int role = Integer.valueOf(selectedRoles[i]);
					/*
					 * Let's see if the role we're assigning is
					 * that of teacher*/	
					 
					/*if ((role == Role.TEACHER) && (!managerUser.isTeacher(user))){
						gotoPage="./loadYearForTeachings";
						
					}*/
					/*
					 * we check if the role we are assigning and'
					 * student*/
					/*if ((role == Role.STUDENT) && (!managerUser.isStudent(user))){
						gotoPage="./showUserList";
						
					} */
					/*
					 * we check if the role we are assigning and'
					 * parent*/
					/*if((role==Role.PARENT)&& (!managerUser.isParent(user))){
						gotoPage="./persistentDataManagement/userManagement/showStudentParentForm.jsp";
					}*/
					managerUser.assignRole(user, role);
				}
			} 
			
			if (unselectedRoles != null) {
				int unselectedlength = unselectedRoles.length;
				for (int i = 0; i < unselectedlength; i++) {
					int role = Integer.valueOf(unselectedRoles[i]);
					if ((managerUser.isAdministrator(user))&&(!itAdmin.hasNext())&&(role==Role.ADMIN)) {
						throw new DeleteManagerException ("Unable to change user role, it is the only Administrator of the system! Create a new Administrator and try again!");
					}
					managerUser.removeRole(user, role);
				}
			}
			
			session.setAttribute("user", user);
		} catch (NumberFormatException numberFormatException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + numberFormatException.getMessage();
			gotoPage = "./error.jsp";
			numberFormatException.printStackTrace();
		} catch (EntityNotFoundException entityNotFoundException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + entityNotFoundException.getMessage();
			gotoPage = "./error.jsp";
			entityNotFoundException.printStackTrace();
		} catch (SQLException SQLException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + SQLException.getMessage();
			gotoPage = "./error.jsp";
			SQLException.printStackTrace();
		} catch (ConnectionException connectionException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + connectionException.getMessage();
			gotoPage = "./error.jsp";
			connectionException.printStackTrace();
		} catch (InvalidValueException invalidValueException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + invalidValueException.getMessage();
			gotoPage = "./error.jsp";
			invalidValueException.printStackTrace();
		} catch (DeleteManagerException deleteManagerException) {
			errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + deleteManagerException.getMessage();
			gotoPage = "./error.jsp";
			deleteManagerException.printStackTrace();
		} catch (IOException ioException) {
			errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
			gotoPage = "./error.jsp";
			ioException.printStackTrace();
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
