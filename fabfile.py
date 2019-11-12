"""Tasks for production deployment using Fabric3 library"""
from fabric.api import task, env, cd, run
from fabric.colors import green, red
from fabric.contrib.console import confirm

# the user to use for the remote commands
env.user = 'gperakis'

# hosts for each operation mode
common_host = 'athena.gr'

env.roledefs = {
    'training': {'hosts': [common_host], 'type': 'gpu'},
    'testing': {'hosts': [common_host], 'type': 'cpu'}}

# git and repository configuration
# Deploy a custom branch by using --set=git_branch=<branch_name> in CLI
if not hasattr(env, 'git_branch'):
    env.git_branch = 'master'
env.git_remote = 'origin'

# project name and directory
env.project_name = 'LGM-SimilarityLearning'
env.working_dir = f'/srv/{env.project_name}'


@task
def pull(force=True, branch=None):
    """Pull latest changes from git repository.
    Parameters
    ----------
    force : bool
        Whether or not to discard any local changes
    branch : str or None
        A git branch name which is used to pull from
    """
    with cd(env.working_dir):
        origin = env.git_remote
        branch = branch if branch else env.git_branch

        if force:
            run('git fetch {} {}'.format(origin, branch))
            run('git reset --hard {}/{}'.format(origin, branch))
        else:
            run('git pull {} {}'.format(origin, branch))
    print(green('Code updated successfully'))


@task
def create_symlinks():
    """Create symbolic links for Pipfile and Pipfile.lock files.

    For each environment we operate, there is a separate list of third-party
    libraries we depend on. So, it is important to point to the appropriate
    Pipfile and Pipfile.lock which contain the corresponding dependencies.
    This is realized by using a symbolic link, which connects the "Pipfile"
    and "Pipfile.lock" default names to the appropriate variation.
    """
    deps_type = env.roledefs[env.effective_roles[0]]['type']

    with cd(env.working_dir):
        run(f'ln -sf Pipfile.{deps_type} Pipfile')
        run(f'ln -sf Pipfile.{deps_type}.lock Pipfile.lock')
    print(green('Symlinks created successfully'))


@task
def update_dependencies(skip_lock: bool = False):
    """
    Update project dependencies using pipenv tool.
    It reads the Pipfile.lock and installs the necessary dependencies.

    Parameters
    ----------
    skip_lock : bool
        Whether we will skip the Pipfile.lock file and we will install directly
        from the Pipfile.
    """
    cmd = 'pipenv install --deploy'
    if skip_lock:
        cmd += ' --skip-lock'

    with cd(env.working_dir):
        run(cmd)
    print(green('Dependencies updated successfully'))


@task
def deploy(force=False):
    """Deploy the latest code master branch.

    Parameters
    ----------
    force : bool
        whether to force pull latest code or not.
    """

    if not env.roles:
        print(
            red("Please choose a correct deployment mode!(training/testing)"))
        return

    if not confirm("Deploying to {}. Are you sure?".format(env.host)):
        print(red("Aborting at user request."))
        return

    pull(force)
    create_symlinks()
    update_dependencies()

    print(green('Successfully deployed on {} VM'.format(env.roles[0])))
