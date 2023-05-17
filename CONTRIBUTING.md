# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
a thread in the [#guild-python-roundtable](https://recursion.slack.com/archives/C04P1FA5YJV) slack channel, or via a [discussion on the `rxrx-python` repository](https://github.com/recursionpharma/rxrx-python/discussions). This is to save everyone time by:
 - ensuring there are no prior/current attempts at something similar, efforts can be consolidated and past pitfalls can be avoided.
 - allowing CODEOWNERS to make suggestions before you start or point you in relevant directions, speeding up development time.
 - helping CODEOWNERS get familiar with the work before it is submitted, hopefully accelerating review time.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Repository Structure

Note, the "template" part of this repository resides in the `{{cookiecutter.repo_name}}` directory, and is likely where all of your edits will take place. Everything outside of that is in service of maintaining this template, and likely only `CODEOWNERS` should need to care about maintaining. That said, as a contributor you will want to understand how to interact with those pieces:

- `hooks` - a directory of hooks that will run as part of the `cookiecutter apply` process. An example would include cleaning up unnecessary files from disabled features.
- `template-tests`
  - `requirements.txt` - The python requirements needed to test this template
  - `test_generate.py` - A script to call `cookiecutter` in a similar fashion to how `roadie` does during a `roadie quickstart` invocation.
  - `test.sh` - A shell script that will exercise this template under different scenarios. The intention here is to mimic all of the prechecks and unit testing provided by the [python/library](https://github.com/recursionpharma/build-pipelines/blob/trunk/python/library.yml) build that new repositories generated from this template will be wired into.
- `CODEOWNERS` - the group of developers responsible for maintaining this template
- `CONTRIBUTING.md` - This file
- `cookiecutter.json` - The specification for what values this template takes. If you modify this to include more values, you will likely need to make a corresponding change in `roadie` to pass values to this.
- `README.md` - Main documentation of what this template provides and how to use it.

## Pull Request Process

1. After creating an issue, a slack thread, or a discussion post, it is recommended that you develop in the open. This means opening a PR in draft mode that any interested parties can see and comment on if they so desire. We use a squash-and-merge strategy with the PR title set to the default commit message. We ask that you employ (conventional commits)[https://www.conventionalcommits.org/en/v1.0.0/] naming convention for PR titles to help signal impact. We have a small pull request template that we ask you to fill out to the best of your ability. Lastly, if you are linking an issue, placing a line like `Closes #21` in your PR description will make it so the linked issue is automatically closed when the PR is merged.
2. When you modify code in a branch, it is highly recommended that you first get the build passing locally before requesting review from the `CODEOWNERS` of this repository on your PR. A PR will run the [eng-tools/template-validator pipeline](https://github.com/recursionpharma/build-pipelines/blob/trunk/eng-tools/template-validator.yml), so you would similarly want to ensure that from the `template-tests` directory a direct invocation of `test.sh` results in a non-failing execution. Once your tests pass locally and/or in our CI system, you can mark your PR as ready for review. See the contract section below for expected response times.
3. A best effort should be made to update the `README.md` where applicable. `CODEOWNERS` should ultimately be responsible for aiding/enforcing this during the review process if it is not handled by the contributor.
4. You may merge the Pull Request in once you have the sign-off of at least one `CODEOWNER`.
5. Refer to the contract section below for obligations of the contributor after a PR is merged.

## Contract

#### CODEOWNERS Service Level Agreement

Upon proper adherence to the steps in the Pull Request Process section above, a `CODEOWNER` will acknowledge receipt and proper labeling and assignment of your PR and asosciated issue within 3 business days of marking the pull request as "Ready." (i.e., not "draft mode"). From here, one or more `CODEOWNER`s will be assigned to shepherd a contributor's PR to completion. This includes suggesting improvements to the changeset, directing the contributor to relevant documentation and areas of the code that could/should be addressed, answering any questions raised by the contributor, providing broader context implied by the changeset, aiding in properly documenting the changeset, and approving and merging the changeset.

#### Contributor 30-Day Warranty

The contributor by submitting a PR that is ultimately merged agrees to provide patches, clarification of usage, and assistance in correcting any regressions caused as a direct or indirect result of their changeset for a period of 30 calendar days after the day their changeset is merged.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Remember our Company Values as you work with others in this project space:

:heart: We Care :heart:
 - Treat each other with respect.

:mortar_board: We Learn :mortar_board:
 - Educate others where appropriate, but also be curious and ask questions when something is not self-evident. Assume nothing.

:envelope: We Deliver :envelope:
 - As a `CODEOWNER`, bias for approval, use "yes, and" or "yes, if" language when suggesting changes.
 - As a contributor, do your best and favor sustainable/durable solutions over quick "this fixes my problem" hacks.

:rocket: We Act Boldly with Integrity :rocket:
 - You are here and contributing, congratulations you are doing this one!

:one: We are One Recursion :one:
 - Be mindful of how your changes could negatively impact other use cases, and open to suggestions from others.
 - Ask for feedback, when you are not sure, and do not limit yourself to only reviewers from the `CODEOWNERS` group. If you know someone who will be impacted or will benefit from your changes, let them know.

### Our Responsibilities

`CODEONWERS` are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

`CODEONWERS` have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
